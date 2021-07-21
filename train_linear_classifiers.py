from functools import reduce
from pathlib import Path
from typing import List

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

n_classes: int = 10

linear_classifier_ckpts: Path = Path("classifier_ckpts")
linear_classifier_ckpts.mkdir(exist_ok=True, parents=True)

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
    'model4': None,
    'model5': None,
}


get_params = lambda tensor: reduce(lambda x, y: x*y, tensor.shape)


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
    validation_dataset_size = min(len(validation_dataset_loader), opt.max_dataset_size_validation)

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
    model1_hook = model.netG.model1.register_forward_hook(get_activation('model1', activation=activation))
    model2_hook = model.netG.model2.register_forward_hook(get_activation('model2', activation=activation))
    model3_hook = model.netG.model3.register_forward_hook(get_activation('model3', activation=activation))
    model4_hook = model.netG.model4.register_forward_hook(get_activation('model4', activation=activation))
    model5_hook = model.netG.model5.register_forward_hook(get_activation('model5', activation=activation))

    # Create separate linear classifiers, one for each layer, independently
    linear_models = {
        'model1': nn.Sequential(nn.Linear(9216, n_classes), nn.Softmax(-1)),
        'model2': nn.Sequential(nn.Linear(9216, n_classes), nn.Softmax(-1)),
        'model3': nn.Sequential(nn.Linear(9216, n_classes), nn.Softmax(-1)),
        'model4': nn.Sequential(nn.Linear(8192, n_classes), nn.Softmax(-1)),
        'model5': nn.Sequential(nn.Linear(8192, n_classes), nn.Softmax(-1))
    }

    # Create optimizers for each linear classifier, independently
    linear_models_optimizers = {
        'model1': torch.optim.SGD(linear_models['model1'].parameters(), lr=1e-3),
        'model2': torch.optim.SGD(linear_models['model2'].parameters(), lr=1e-3),
        'model3': torch.optim.SGD(linear_models['model3'].parameters(), lr=1e-3),
        'model4': torch.optim.SGD(linear_models['model4'].parameters(), lr=1e-3),
        'model5': torch.optim.SGD(linear_models['model5'].parameters(), lr=1e-3),
    }

    # Keep track of validation losses for all linear classifiers, independently
    linear_models_validation_losses = {
        'model1': -np.inf,
        'model2': -np.inf,
        'model3': -np.inf,
        'model4': -np.inf,
        'model5': -np.inf,
    }

    loss_fn = nn.CrossEntropyLoss()

    # Writer will output to ./runs/ directory by default
    writer = SummaryWriter()

    # TODO extract all validation data into memory?
    validation_batches = get_validation_feature_tensors(
        activation=activation, device=device, model=model, opt=opt,
        validation_dataset_loader=validation_dataset_loader,
        validation_dataset_size=validation_dataset_size
    )

    # Loop over epochs
    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay):

        # Place all models in "train" mode
        for _, linear_model in linear_models.items():
            linear_model.train()

        # Variables to keep track of metrics
        training_accuracy: List[float] = []
        training_loss: List[float] = []

        # for i, data in enumerate(dataset):
        for i, data_raw in tqdm(enumerate(dataset_loader), total=dataset_size // opt.batch_size):

            # Run data through model to get feature activations
            with torch.no_grad():
                data_raw[0] = data_raw[0].to(device)
                data = util.get_colorization_data(data_raw, opt, p=opt.sample_p)
                if data is None: continue
                model.set_input(data)
                model.test(compute_losses=False)
                outputs = reshape_activation_outputs(activation)

            # Loop through individual linear classifiers
            for model_name, linear_classifier in linear_models.items():
                # Get linear model predictions
                preds = linear_classifier(outputs[model_name])
                # Calculate CE loss
                loss = loss_fn(preds, data_raw[1])
                # Optimize and Backpropagation
                optimizer = linear_models_optimizers[model_name]
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_acc = torch.sum(preds.argmax(-1) == data_raw[1]) / preds.shape[-1]

                training_accuracy.append(train_acc.detach())
                training_loss.append(loss.detach())

            # # time to load data
            # iter_start_time = time.time()
            # if total_steps % opt.print_freq == 0:
            #     t_data = iter_start_time - iter_data_time
            #
            # # visualizer.reset()
            # total_steps += opt.batch_size
            # epoch_iter += opt.batch_size
            # model.set_input(data)
            # model.optimize_parameters()
            #
            # if total_steps % opt.display_freq == 0:
            #     save_result = total_steps % opt.update_html_freq == 0
            #     # visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)
            #
            # # time to do forward & backward
            # if total_steps % opt.print_freq == 0:
            #     losses = model.get_current_losses()
            #     t = time.time() - iter_start_time
            #     # visualizer.print_current_losses(epoch, epoch_iter, losses, t, t_data)
            #     # if opt.display_id > 0:
            #     # visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, opt, losses)
            #
            # if total_steps % opt.save_latest_freq == 0:
            #     print('saving the latest model (epoch %d, total_steps %d)' %
            #           (epoch, total_steps))
            #     model.save_networks('latest')
            #
            # iter_data_time = time.time()

            # Break inner loop if theres enough data
            if ((i + 1) * opt.batch_size) >= opt.max_dataset_size:
                break

        # Add training metrics to graph
        writer.add_scalar(f'TrainLoss/{model_name}', sum(training_loss)/len(training_loss), epoch)
        writer.add_scalar(f'TrainAccuracy/{model_name}', sum(training_accuracy)/len(training_accuracy), epoch)
        # writer.close()

        # Evaluate the classifier performance on validation data
        with torch.no_grad():

            for model_name, linear_model in linear_models.items():
                model.eval()

                losses = 0
                accuracies = 0
                for validation_data_raw in validation_batches:
                    xv, yv = validation_data_raw
                    output = linear_model(xv)
                    loss = loss_fn(output, yv)
                    acc = torch.sum(output.argmax(-1) == yv) / output.shape[-1]
                    losses += loss
                    accuracies += acc

                # Add training metrics to graph
                validation_loss = losses / len(validation_batches)
                writer.add_scalar(f'ValLoss/{model_name}', validation_loss, epoch)

                # Current lowest validation loss
                curr_validation_loss = linear_models_validation_losses[model_name]

                # Check if best model or not
                if validation_loss <= curr_validation_loss:

                    # This is best validation loss so far, save model as "best" and update dictionary
                    print(f"Model {model_name} best validation loss so far: {validation_loss} < {curr_validation_loss}")
                    torch.save(linear_model.state_dict(), f"{linear_classifier_ckpts}/{model_name}_best.pth")
                    linear_models_validation_losses[model_name] = validation_loss

                # Record validation accuracy
                validation_accuracy = accuracies / len(validation_batches)
                writer.add_scalar(f'ValAccuracy/{model_name}', validation_accuracy, epoch)

                # Always save checkpoints at every epoch
                torch.save(linear_model.state_dict(), f"{linear_classifier_ckpts}/{model_name}_e{epoch}.pth")

    # Close tensorboard writer
    writer.close()


if __name__ == '__main__':

    opt = TrainOptions().parse()

    opt.dataroot = opt.dataroot if opt.dataroot is not None else './dataset/ilsvrc2012/%s/' % opt.phase
    assert opt.dataroot_validation is not None, "When training linear classifiers, please specify a validation " \
                                                "dataset via --dataroot_validation"

    main(opt=opt)
