# This is the script used to train the linear classifiers on top of the model layer feature outputs.
# Training and Validation data must be set up in a nested format, appropriate for object classification.
# For example: "./dataset/train/<CLASS>/SOME_IMAGE.JPEG

CHECKPOINTS_DIR="./checkpoints"

DATAROOT="PATH/TO/TRAINING/DATA"                # <-- Set to training data path (i.e, "./dataset/ilsvrc2012/train")
DATAROOT_VALIDATION="PATH/TO/VALIDATION/DATA"   # <-- Set to validation data (i.e, "./dataset/ilsvrc2012/val")

COCO_DATAROOT="PATH/TO/TRAINING/DATA"               # <-- (COCO) Set to training data path (i.e, "./dataset/ilsvrc2012/train")
COCO_DATAROOT_VALIDATION="PATH/TO/VALIDATION/DATA"  # <-- (COCO) Set to validation data (i.e, "./dataset/ilsvrc2012/val")

MAX_DATASET_SIZE=13000                              # <-- This is per epoch


# CAFFE PRETRAINED
python train_linear_classifiers.py \
  --name siggraph_caffemodel \
  --checkpoints_dir="$CHECKPOINTS_DIR" \
  --linear_checkpoints="classifier_ckpts_pretrained" \
  --gpu_ids 0 \
  --dataroot="$DATAROOT" \
  --dataroot_validation="$DATAROOT_VALIDATION" \
  --max_dataset_size=$MAX_DATASET_SIZE \
  --niter=200 \
  --niter_decay 0 \
  --mask_cent 0

# TRAINED MODEL
python train_linear_classifiers.py --name siggraph_reg2_tom \
  --checkpoints_dir="$CHECKPOINTS_DIR" \
  --linear_checkpoints="classifier_ckpts_modified" \
  --gpu_ids 0 \
  --dataroot="$DATAROOT" \
  --dataroot_validation="$DATAROOT_VALIDATION" \
  --max_dataset_size=$MAX_DATASET_SIZE \
  --niter=200 \
  --niter_decay 0

# STYLIZED MODEL
python train_linear_classifiers.py --name siggraph_reg2 \
  --checkpoints_dir="$CHECKPOINTS_DIR" \
  --linear_checkpoints="classifier_ckpts_stylized" \
  --gpu_ids 0 \
  --dataroot="$DATAROOT" \
  --dataroot_validation="$DATAROOT_VALIDATION" \
  --max_dataset_size=$MAX_DATASET_SIZE \
  --niter=200 \
  --niter_decay 0

# COCO MODEL
python train_linear_classifiers.py --name siggraph_reg2 \
  --checkpoints_dir="$CHECKPOINTS_DIR" \
  --linear_checkpoints="classifier_ckpts_coco" \
  --gpu_ids 0 \
  --dataroot="$COCO_DATAROOT" \
  --dataroot_validation="$COCO_DATAROOT_VALIDATION" \
  --max_dataset_size=$MAX_DATASET_SIZE \
  --niter=200 \
  --niter_decay 0
