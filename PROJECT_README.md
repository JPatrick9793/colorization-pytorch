# CS7643 DEEP LEARNING PROJECT (GROUP 170) - COLORIZING BLACK AND WHITE IMAGES

## Description
***

This repository is a fork of [Colorful Image Colorization](https://github.com/richzhang/colorization) [[1]](#1).
For this project, we retrained the siggraph model using three modifications, 
with the goal of conducting intrinsic evaluation on colorization performance:
1. Modification of the Loss function, 
   to encourage the learner to choose values producing a higher saturation effect: 
    - Original loss function: `loss_G = loss_G_CE * l + loss_G_L1_reg` 
      (where: l is an adjustable parameter).
    - Modified loss function: `loss_G = loss_G_CE * l + loss_G_L1_reg + 1e-4 * S_reg` 
      (where: S_reg is a penalty term based on average saturation of predicted images).
2. Train the siggraph model, but using a [stylized](https://github.com/bethgelab/stylize-datasets) version of the ImageNet dataset. 
    The goal is to evaluate the importance of texture vs. shape bias for re-colorization[[2]](#2). 
    Also, linear classifiers are trained on the first five layers of each model, 
    to evaluate the ability to generalize with respect to object detection.
3. Training the model using the [coco](https://cocodataset.org/#home) dataset, 
   and evaluating the colorization performance against the original model. coco was created not only for 
object detection, but also image segmentation and several other features.

## Dataset
***

For the ImageNet dataset, due to time and computational constraints, we reduced the full-sized training set down
from 1000 classes and over 2 million images, to 100 classes and 130k images.

- For ImageNet, download from here [here](https://image-net.org/download.php) (you may need to make an account).
- For coco dataset, download from [here](https://cocodataset.org/#home).
- To create a stylized version of ImageNet (or any arbitrary dataset), first download the 
  [Painter by Numbers](https://www.kaggle.com/c/painter-by-numbers) dataset from kaggle, and then follow the 
  instructions presented in this [repo](https://github.com/bethgelab/stylize-datasets).

When retraining the **models for re-colorization**, the training set for all datasets should be in the same format
(e.g., `/PATH_TO_DIRECTORY/<train>/<class>/SOMEIMAGE.JPEG`. The validation data must be in a slightly different
format: `/PATH_TO_DIRECTORY/<val>/SOMEIMAGE.JPEG`. Datasets must also be in this form when testing re-colorization
performance as well. If using ImageNet, you can follow the instructions in the original README, and use the
helper script: `python make_ilsvrc_dataset.py --in_path /PATH/TO/ILSVRC12`

If training the **linear classifiers**, the validation data must be further nested to include class label
(e.g., `/PATH_TO_DIRECTORY/<val>/<class>/SOMEIMAGE.JPEG`).

## How to install
***

We use the same environment as in the original repository, so install using `pip install -r requirements.txt`. 
A summary of the packages is below:
```
pip install torch==0.4.1.post2
pip install torchvision==0.2.1
pip install scipy==1.1.0
pip install dominate
```

You can also run this using the Colab-pro: GPU, high-memory platform.

## Instructions
***
### Checkpoints
Code from the original repository has been modified slightly to account for bug-fixes as well as additional features.
To download the original pretrained caffe model, follow the instructions in the original README and run 
`bash pretrained_models/download_siggraph_model.sh`. This will put the model weights under the "checkpoints" folder.
This folder is also where coco, stylized, and modified checkpoints will be stored. Under this "checkpoints" folder
are a set of nested folders, each one corresponding to the model they are associated with

### Training the siggraph models (In General)
Generally, we follow the instructions in the original README and run `bash ./scripts/train_siggraph.sh`; however, 
you will need to make slight modifications to this script depending on which tests you are running. Training is
done in 4 phases: The first and seconds phases train the siggraph model on classification loss, however, the 
first phase only uses a small subset of the overall training data (almost like a test). The third and fourth 
phases then train the model using regression loss (this is where we inject our "modified" loss logic, if applicable).
The difference between the third and fourth phase is that the learning rate is decreased on the fourth phase. In 
other words, it's similar to manually scheduling the learning rate decay by hand.

train.py has several flags, many of which are not necessary for replication, but they can be found in 
`options/base_options.py` and `options/train_options.py`. The main flags used in `./scripts/train_siggraph.sh` are
as follows:
- `--name` — This corresponds to the folder name under the "checkpoints" directory. 
  For example, Phase 4 training will output to a "siggraph_reg2" checkpoints folder. 
- `--sample_p` — Corresponds to geometric sampling probability of giving the model color hints during training. 
  Phases 1 and 2 do not utilize hints, so this is set to 1.0. The third and fourth phases set this to 0.125. 
- `--niter` — This corresponds to how many epochs to run the model for. Each epoch runs through the entire dataset (or
  under max_dataset_size is reached).
- `--niter_decay` — This many epochs after "niter" are added to training; 
  however, the learning rate is linearly decayed to 0 over these epochs.
- `--classification` — If this flag is set, train the model using "classification" loss, 
  otherwise default to regression.
- `--load_model` — If this flag is set, load the model from the `latest_net_G.pth` checkpoint file.
  Note, before the second, third, and fourth phases of training, the `latest_net_G.pth` file is copied over
  from the previous phase, so we can continue training.
- `--phase` — For training, this will always be set to 'train'.
- `--dataroot` — (optional) This is a modification of the original repo, and allows us to point to any location
  for the dataset. If not specified, it will default to `./dataset/ilsvrc2012/<phase>`, where 'phase' corresponds
  to the previous flag.
- `--checkpoints_dir` — (optional) Another modification to the original repo, this allows us to specify an arbitrary
  folder to use for checkpoint directory. If not specified, this defaults to `./checkpoints`
- `--max_dataset_size` — (optional) Another modification to the original repo. If specified, the training epoch will
  stop after seeing this many images. For example, the value '130000' is set during stylized training to match 
  the training volume of the other models.
- `--mask_cent 0` (Optional) Only use this when using the pretrained caffe model.
- `--gpu_ids` (Optional) Which GPU to use for training. Set to `-1` if using CPU.

### Training siggraph model with modified loss function

TODO - are there any extra files/code that needs to be added for this?

### Training the Linear Classifiers

In order to train the linear classifiers, we have created a helper script: `bash train_linear_classifiers.sh`. 
This script calls the new `train_linear_classifiers.py` file we have created. The script runs through
all four different types of models: Pretrained, Modified loss function, stylized imagenet, and coco models. 
Many of the flags are the same as the original training script, however, there are a few differences:
- `--name` — This still corresponds to the folder name under the "checkpoints" directory, and determines which
  siggraph model is being used.
- `--checkpoints_dir` — (optional) Another modification to the original repo, this allows us to specify an arbitrary
  folder to use for checkpoint directory. If not specified, this defaults to `./checkpoints`
- `--linear_checkpoints` — This determines the name/location of where the linear classifier checkpoints are saved.
- `--gpu_ids` (Optional) Which GPU to use for training. Set to `-1` if using CPU.
- `--dataroot` — (optional) This is a modification of the original repo, and allows us to point to any location
  for the dataset. If not specified, it will default to `./dataset/ilsvrc2012/train`.
- `--dataroot_validation` — (optional) This is a modification of the original repo, 
  and allows us to point to any location for the validation dataset. 
  If not specified, it will default to `./dataset/ilsvrc2012/val`.
- `--max_dataset_size` — (optional) Another modification to the original repo. If specified, the training epoch will
  stop after seeing this many images. For example, the value '130000' is set during stylized training to match 
  the training volume of the other models.
- `--niter` — This corresponds to how many epochs to run the model for. Each epoch runs through the entire dataset 
  (or under max_dataset_size is reached).
- `--mask_cent 0` (Optional) Only use this when using the pretrained caffe model.

## References
<a id="1">[1]</a>
Zhang, R., Zhu, J. Y., Isola, P., Geng, X., Lin, A. S., Yu, T., & Efros, A. A. (2017). 
Real-time user-guided image colorization with learned deep priors. 
arXiv preprint arXiv:1705.02999.

<a id="2"[2]</a>
Geirhos, Robert, et al. 
"ImageNet-trained CNNs are biased towards texture; increasing shape bias improves accuracy and robustness." 
arXiv preprint arXiv:1811.12231 (2018).