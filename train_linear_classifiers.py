from functools import reduce
from pathlib import Path
from typing import List, Dict

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models import create_model
from options.train_options import TrainOptions
from util import util
from sklearn.metrics import accuracy_score, f1_score

n_classes: int = 100

kernel_sizes = {
    'model1': 32,
    'model2': 16,
    'model3': 16,
    'model4': 8,
    'model5': 8,
}

interpolate_size = {
    'model1': (12, 12),
    'model2': (9, 8),
    'model3': (6, 6),
    'model4': (4, 4),
    'model5': (4, 4),
}

get_params = lambda tensor: reduce(lambda x, y: x * y, tensor.shape)


def reshape_activation_outputs(activation):
    # global kernel_sizes
    # global interpolate_size

    outputs = {}
    for key, items in activation.items():

        # Acquire dimensions
        batch, depth, width, height = items.shape

        # Get kernel size
        kernel_size = kernel_sizes[key]

        # Pool the tensor
        output = F.avg_pool2d(items, kernel_size=kernel_size, stride=kernel_size)

        interp_size = interpolate_size[key]
        if interp_size is not None:
            output = F.interpolate(input=output, size=interp_size, scale_factor=None,
                                   mode='bilinear', align_corners=True)

        outputs[key] = output.view(batch, -1)

    return outputs


def get_validation_feature_tensors(activation, device, model, opt, validation_dataset_loader, validation_dataset_size):
    validation_batches = []
    with torch.no_grad():
        for e, data_raw in tqdm(enumerate(validation_dataset_loader), total=validation_dataset_size // opt.batch_size):
            data_raw[0] = data_raw[0].to(device)
            data = util.get_colorization_data(data_raw, opt, p=opt.sample_p)
            if data is None: continue
            model.set_input(data)
            model.test(compute_losses=False)
            outputs = reshape_activation_outputs(activation)
            validation_batches.append(outputs)

            # Break inner loop if theres enough data
            if ((e + 1) * opt.batch_size) >= opt.max_dataset_size:
                break

    return validation_batches


def get_dataloader(dataset, opt, shuffle: bool = True):
    dataset_loader = torch.utils.data.DataLoader(
        dataset, batch_size=opt.batch_size, shuffle=shuffle, num_workers=int(opt.num_threads))
    return dataset_loader


def get_dataset(opt, dataroot):
    dataset = torchvision.datasets.ImageFolder(dataroot,
                                               transform=transforms.Compose([
                                                   transforms.RandomChoice([
                                                       transforms.Resize(opt.loadSize, interpolation=1),
                                                       transforms.Resize(opt.loadSize, interpolation=2),
                                                       transforms.Resize(opt.loadSize, interpolation=3),
                                                       transforms.Resize((opt.loadSize, opt.loadSize), interpolation=1),
                                                       transforms.Resize((opt.loadSize, opt.loadSize), interpolation=2),
                                                       transforms.Resize((opt.loadSize, opt.loadSize), interpolation=3)
                                                   ]),
                                                   transforms.RandomChoice([
                                                       transforms.RandomResizedCrop(opt.fineSize, interpolation=1),
                                                       transforms.RandomResizedCrop(opt.fineSize, interpolation=2),
                                                       transforms.RandomResizedCrop(opt.fineSize, interpolation=3)
                                                   ]),
                                                   transforms.RandomHorizontalFlip(),
                                                   transforms.ToTensor()]))
    return dataset


def main(opt):

    # Create checkpoints
    # linear_classifier_ckpts: Path = Path("classifier_ckpts_pretrained")
    assert opt.linear_checkpoints is not None, "Please specify output directory for checkpoints"
    linear_classifier_ckpts: Path = Path(opt.linear_checkpoints)
    linear_classifier_ckpts.mkdir(exist_ok=True, parents=True)

    # Always force load model
    opt.load_model = True

    # Specify CUDA device if passed into args
    device = torch.device("cpu" if len(opt.gpu_ids) <= 0 else f"cuda:{opt.gpu_ids[0]}")

    # Load training data
    print(f"Creating Training Dataset Loader")
    dataset = get_dataset(opt=opt, dataroot=opt.dataroot)
    dataset_loader = get_dataloader(dataset, opt)
    dataset_size = min(len(dataset), opt.max_dataset_size)

    # Load validation Data
    print(f"Creating Validation Dataset Loader")
    validation_dataset = get_dataset(opt=opt, dataroot=opt.dataroot_validation)
    validation_dataset_loader = get_dataloader(dataset=validation_dataset, opt=opt, shuffle=False)
    validation_dataset_size = min(len(validation_dataset), opt.max_dataset_size_validation)

    # Load siggraph model for feature extraction
    print('#training images = %d' % dataset_size)
    model = create_model(opt)
    model.setup(opt)
    model.eval()

    # TODO does passing in "activation" work by reference, and does it help?
    # Wrapper function to create "hooks" for extracting layer activations
    activation = {}

    def get_activation(name, activation):
        def hook(model, input, output):
            activation[name] = output.detach()

        return hook

    # Place hooks in original model to extract the features at each layer
    model1_hook = model.netG.module.model1.register_forward_hook(get_activation('model1', activation=activation))
    model2_hook = model.netG.module.model2.register_forward_hook(get_activation('model2', activation=activation))
    model3_hook = model.netG.module.model3.register_forward_hook(get_activation('model3', activation=activation))
    model4_hook = model.netG.module.model4.register_forward_hook(get_activation('model4', activation=activation))
    model5_hook = model.netG.module.model5.register_forward_hook(get_activation('model5', activation=activation))

    # Create separate linear classifiers, one for each layer, independently
    linear_models = {
        'model1': nn.Sequential(nn.Linear(9216, n_classes), nn.Softmax(-1)).to(device),
        'model2': nn.Sequential(nn.Linear(9216, n_classes), nn.Softmax(-1)).to(device),
        'model3': nn.Sequential(nn.Linear(9216, n_classes), nn.Softmax(-1)).to(device),
        'model4': nn.Sequential(nn.Linear(8192, n_classes), nn.Softmax(-1)).to(device),
        'model5': nn.Sequential(nn.Linear(8192, n_classes), nn.Softmax(-1)).to(device)
    }

    # Create optimizers for each linear classifier, independently
    linear_models_optimizers = {
        'model1': torch.optim.Adam(linear_models['model1'].parameters(), lr=1e-3),
        'model2': torch.optim.Adam(linear_models['model2'].parameters(), lr=1e-3),
        'model3': torch.optim.Adam(linear_models['model3'].parameters(), lr=1e-3),
        'model4': torch.optim.Adam(linear_models['model4'].parameters(), lr=1e-3),
        'model5': torch.optim.Adam(linear_models['model5'].parameters(), lr=1e-3),
    }

    # Keep track of validation losses for all linear classifiers, independently
    linear_models_validation_losses = {
        'model1': np.inf,
        'model2': np.inf,
        'model3': np.inf,
        'model4': np.inf,
        'model5': np.inf,
    }

    loss_fn = nn.CrossEntropyLoss()

    # Writer will output to ./runs/ directory by default
    writer = SummaryWriter()

    # Loop over epochs
    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay):

        # Place all models in "train" mode
        for _, linear_model in linear_models.items():
            linear_model.train()

        # Variables to keep track of metrics
        training_losses: Dict[str, List[float]] = {model_name: [0.0] for model_name in linear_models}
        training_accuracies: Dict[str, List[float]] = {model_name: [0.0] for model_name in linear_models}
        training_f1_scores: Dict[str, List[float]] = {model_name: [0.0] for model_name in linear_models}

        # for i, data in enumerate(dataset)
        for i, data_raw in tqdm(enumerate(dataset_loader), total=dataset_size // opt.batch_size):

            # Place data on GPU (or CPU)
            data_raw[0] = data_raw[0].to(device)
            data_raw[1] = data_raw[1].to(device)

            # Run input batch through model to get feature activations
            with torch.no_grad():
                data = util.get_colorization_data(data_raw, opt, p=opt.sample_p)
                if data is None:
                    continue
                model.set_input(data)
                model.test(compute_losses=False)
                outputs = reshape_activation_outputs(activation)

            # Sometimes batch sizes don't line up, so skip this batch if that happens...
            if outputs['model1'].shape[0] != data_raw[1].shape[0]:
                # print('xv.shape[0] != yv[1].shape[0]')
                continue

            # Loop through individual linear classifiers per batch
            for model_name, linear_classifier in linear_models.items():

                # Run feature vector through linear classifier
                feature_input = outputs[model_name]
                preds = linear_classifier(feature_input)

                # Optimize and Backpropagation
                loss = loss_fn(preds, data_raw[1])
                optimizer = linear_models_optimizers[model_name]
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Find batch metrics, and add to list
                train_accuracy, train_f1 = get_numpy_metrics(targets=data_raw[1], preds=preds)
                training_accuracies[model_name].append(train_accuracy)
                training_losses[model_name].append(loss.detach())
                training_f1_scores[model_name].append(train_f1)

            # Break inner loop if theres enough data
            if ((i + 1) * opt.batch_size) >= opt.max_dataset_size:
                break

        # Iterate over linear models at the end of the epoch to add training metrics to tensorboard
        for model_name, linear_model in linear_models.items():
            writer.add_scalar(f'TrainAccuracy/{model_name}',
                              sum(training_accuracies[model_name]) / len(training_accuracies[model_name]), epoch)
            writer.add_scalar(f'TrainLoss/{model_name}',
                              sum(training_losses[model_name]) / len(training_losses[model_name]), epoch)
            writer.add_scalar(f'TrainF1/{model_name}',
                              sum(training_f1_scores[model_name]) / len(training_f1_scores[model_name]), epoch)

        # Evaluate the classifier performance on validation data
        with torch.no_grad():

            # Convert all linear classifiers to eval mode
            for model_name, linear_model in linear_models.items():
                model.eval()

            validation_losses: Dict[str, List[float]] = {model_name: [0.0] for model_name in linear_models}
            validation_accuracies: Dict[str, List[float]] = {model_name: [0.0] for model_name in linear_models}
            validation_f1_scores: Dict[str, List[float]] = {model_name: [0.0] for model_name in linear_models}

            # iterate over validation data
            for e, data_raw in tqdm(enumerate(validation_dataset_loader),
                                    total=validation_dataset_size // opt.batch_size):

                # Run image through model for feature vectors
                data_raw[0] = data_raw[0].to(device)
                data_raw[1] = data_raw[1].to(device)
                data = util.get_colorization_data(data_raw, opt, p=opt.sample_p)
                if data is None: continue
                model.set_input(data)
                model.test(compute_losses=False)
                outputs = reshape_activation_outputs(activation)

                # Skip this validation batch if there was an error during processing
                if outputs[model_name].shape[0] != data_raw[1].shape[0]:
                    # print('xv.shape[0] != yv[1].shape[0]')
                    continue

                # Iterate over linear classifiers to evaluate validation batch
                for model_name, linear_model in linear_models.items():
                    val_feature_vector = outputs[model_name]
                    val_preds = linear_model(val_feature_vector)
                    loss = loss_fn(val_preds, data_raw[1])

                    # Add accuracy and loss to dictionary to keep track
                    accuracy, f1 = get_numpy_metrics(targets=data_raw[1], preds=val_preds)
                    validation_losses[model_name].append(loss)
                    validation_accuracies[model_name].append(accuracy)
                    validation_f1_scores[model_name].append(f1)

                # Break inner loop if theres enough data
                if ((e + 1) * opt.batch_size) >= opt.max_dataset_size:
                    break

            # Iterate over models one final time to evaluate overall validation performance for this epoch
            for model_name, linear_model in linear_models.items():

                print(f"Evaluating {model_name}")

                # Current lowest validation loss for this linear classifier
                curr_validation_loss = linear_models_validation_losses[model_name]

                # Add validation metrics to graph
                validation_loss = sum(validation_losses[model_name]) / len(validation_losses[model_name])
                validation_accuracy = sum(validation_accuracies[model_name]) / len(validation_accuracies[model_name])
                validation_f1 = sum(validation_f1_scores[model_name]) / len(validation_f1_scores[model_name])
                writer.add_scalar(f'ValLoss/{model_name}', validation_loss, epoch)
                writer.add_scalar(f'ValAccuracy/{model_name}', validation_accuracy, epoch)
                writer.add_scalar(f'ValF1/{model_name}', validation_f1, epoch)

                # Check if best model or not
                if validation_loss <= curr_validation_loss:

                    # This is best validation loss so far, save model as "best" and update dictionary
                    print(f"Model {model_name} best validation loss so far: {validation_loss} < {curr_validation_loss}")
                    torch.save(linear_model.state_dict(), f"{linear_classifier_ckpts}/{model_name}_best.pth")
                    linear_models_validation_losses[model_name] = validation_loss

                else:
                    print(f"Model {model_name} validation did not improve: {validation_loss} > {curr_validation_loss}")

                # Always save checkpoints at every epoch
                torch.save(linear_model.state_dict(), f"{linear_classifier_ckpts}/{model_name}_e{epoch}.pth")

    # Close tensorboard writer
    writer.close()


def get_numpy_metrics(targets, preds):
    preds_numpy = preds.argmax(-1).cpu().numpy()
    targt_numpy = targets.cpu().numpy()
    accuracy = accuracy_score(y_pred=preds_numpy, y_true=targt_numpy)
    f1 = f1_score(y_pred=preds_numpy, y_true=targt_numpy, average='micro')
    return accuracy, f1


if __name__ == '__main__':
    opt = TrainOptions().parse()

    opt.dataroot = opt.dataroot if opt.dataroot is not None else './dataset/ilsvrc2012/%s/' % opt.phase
    assert opt.dataroot_validation is not None, "When training linear classifiers, please specify a validation " \
                                                "dataset via --dataroot_validation"

    main(opt=opt)
