# TODO modify this file to point to

# Train classification network on small training set first
python train.py --name siggraph_class_small --sample_p 1.0 --niter 100 --niter_decay 0 --classification \
  --phase train_small \
  --checkpoints_dir="/media/john/New Volume/DLProject/checkpoints" \
  --dataroot="/media/john/New Volume/ILSVRC2012_img_train/train_small_stylized" \
  --max_dataset_size=130000 \

# Train classification network first
mkdir ./checkpoints/siggraph_class
cp ./checkpoints/siggraph_class_small/latest_net_G.pth ./checkpoints/siggraph_class/
mkdir "/media/john/New Volume/DLProject/checkpoints/siggraph_class"
cp "/media/john/New Volume/DLProject/checkpoints/siggraph_class_small/latest_net_G.pth" "/media/john/New Volume/DLProject/checkpoints/siggraph_class"
python train.py --name siggraph_class --sample_p 1.0 --niter 15 --niter_decay 0 --classification \
  --load_model \
  --phase train \
  --dataroot="/media/john/New Volume/DLProject/tom_data/stylized/train" \
  --checkpoints_dir="/media/john/New Volume/DLProject/checkpoints" \
  --max_dataset_size 130000


## Train regression model (with color hints)
#mkdir ./checkpoints/siggraph_reg
#cp ./checkpoints/siggraph_class/latest_net_G.pth ./checkpoints/siggraph_reg/
mkdir "/media/john/New Volume/DLProject/checkpoints/siggraph_reg"
cp "/media/john/New Volume/DLProject/checkpoints/siggraph_class/latest_net_G.pth" "/media/john/New Volume/DLProject/checkpoints/siggraph_reg"
python train.py --name siggraph_reg --sample_p .125 --niter 10 --niter_decay 0 --lr 0.00001 \
  --load_model \
  --phase train  \
  --dataroot="/media/john/New Volume/DLProject/tom_data/stylized/train" \
  --checkpoints_dir="/media/john/New Volume/DLProject/checkpoints" \
  --max_dataset_size 130000

## Turn down learning rate to 1e-6
#mkdir ./checkpoints/siggraph_reg2
#cp ./checkpoints/siggraph_reg/latest_net_G.pth ./checkpoints/siggraph_reg2/
mkdir "/media/john/New Volume/DLProject/checkpoints/siggraph_reg2"
cp "/media/john/New Volume/DLProject/checkpoints/siggraph_reg/latest_net_G.pth" "/media/john/New Volume/DLProject/checkpoints/siggraph_reg2"
python train.py --name siggraph_reg2 --sample_p .125 --niter 5 --niter_decay 0 --lr 0.000001 \
  --load_model \
  --phase train \
  --dataroot="/media/john/New Volume/DLProject/tom_data/stylized/train" \
  --checkpoints_dir="/media/john/New Volume/DLProject/checkpoints" \
  --max_dataset_size 130000
