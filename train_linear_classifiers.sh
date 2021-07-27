DATAROOT="/media/john/New Volume/DLProject/tom_data/train"
DATAROOT_VALIDATION="/media/john/New Volume/ILSVRC2012_img_train/val_nested"
MAX_DATASET_SIZE=13000


## CAFFE PRETRAINED
#python train_linear_classifiers.py --name siggraph_caffemodel \
#  --checkpoints_dir="./checkpoints" \
#  --linear_checkpoints="classifier_ckpts_pretrained" \
#  --gpu_ids 0 \
#  --dataroot="$DATAROOT" \
#  --dataroot_validation="$DATAROOT_VALIDATION" \
#  --max_dataset_size=$MAX_DATASET_SIZE \
#  --niter=200 \
#  --niter_decay 0 \
#  --mask_cent 0
#
#
## TOM TRAINED MODEL
#python train_linear_classifiers.py --name siggraph_reg2_tom \
#  --checkpoints_dir="/media/john/New Volume/DLProject/checkpoints" \
#  --linear_checkpoints="classifier_ckpts_tom" \
#  --gpu_ids 0 \
#  --dataroot="$DATAROOT" \
#  --dataroot_validation="$DATAROOT_VALIDATION" \
#  --max_dataset_size=$MAX_DATASET_SIZE \
#  --niter=200 \
#  --niter_decay 0
#
## STYLIZED MODEL
#python train_linear_classifiers.py --name siggraph_reg2 \
#  --checkpoints_dir="/media/john/New Volume/DLProject/checkpoints" \
#  --linear_checkpoints="classifier_ckpts_stylized" \
#  --gpu_ids 0 \
#  --dataroot="$DATAROOT" \
#  --dataroot_validation="$DATAROOT_VALIDATION" \
#  --max_dataset_size=$MAX_DATASET_SIZE \
#  --niter=200 \
#  --niter_decay 0

# WEICHAO COCO MODEL
python train_linear_classifiers.py --name siggraph_reg2 \
  --checkpoints_dir="/media/john/New Volume/DLProject/weichao_data" \
  --linear_checkpoints="classifier_ckpts_coco" \
  --gpu_ids 0 \
  --dataroot="$DATAROOT" \
  --dataroot_validation="$DATAROOT_VALIDATION" \
  --max_dataset_size=$MAX_DATASET_SIZE \
  --niter=200 \
  --niter_decay 0

