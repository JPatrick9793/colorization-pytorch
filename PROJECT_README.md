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

### Training the siggraph models (In General)
...

### Training siggraph model with modified loss function
...

### Training the Linear Classifiers
...

## References
<a id="1">[1]</a>
Zhang, R., Zhu, J. Y., Isola, P., Geng, X., Lin, A. S., Yu, T., & Efros, A. A. (2017). 
Real-time user-guided image colorization with learned deep priors. 
arXiv preprint arXiv:1705.02999.

<a id="2"[2]</a>
Geirhos, Robert, et al. 
"ImageNet-trained CNNs are biased towards texture; increasing shape bias improves accuracy and robustness." 
arXiv preprint arXiv:1811.12231 (2018).