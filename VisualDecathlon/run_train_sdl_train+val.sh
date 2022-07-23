function train_fn {
    CUDA_VISIBLE_DEVICES=0 python train_sdl.py --dataset $1 --trainval --wd 1. --mode bn --expdir ./results/sdl_train+val/ --datadir ./data/decathlon-1.0/ --imdbdir ./data/decathlon-1.0/annotations/
}

function finetune_fn {
    CUDA_VISIBLE_DEVICES=0 python finetune_sdl.py --dataset $1 --trainval --wd 5. --lr $2 --mode bn --expdir ./results/sdl_train+val/ --datadir ./data/decathlon-1.0/ --imdbdir ./data/decathlon-1.0/annotations/ --source ./results/sdl_train+val/checkpoint/ckptbnimagenet12_best.pth
}
# Train an independent feature extractor on every training dataset (the following models could be trained in parallel)
# setting --trainval to train with train and validation data

# ImageNet
NAME="imagenet12"
train_fn $NAME 

# finetune for other tasks

# Aircraft
NAME="aircraft"; LR=1e-1
finetune_fn $NAME $LR

# cifar100
NAME="cifar100"; LR=1e-2
finetune_fn $NAME $LR

# daimlerpedcls
NAME="daimlerpedcls"; LR=1e-2
finetune_fn $NAME $LR

# dtd
NAME="dtd"; LR=1e-2
finetune_fn $NAME $LR

# gtsrb
NAME="gtsrb"; LR=1e-2
finetune_fn $NAME $LR

# omniglot
NAME="omniglot"; LR=1e-1
finetune_fn $NAME $LR

# svhn
NAME="svhn"; LR=1e-1
finetune_fn $NAME $LR

# ucf101
NAME="ucf101"; LR=1e-1
finetune_fn $NAME $LR

# vgg-flowers
NAME="vgg-flowers"; LR=1e-1
finetune_fn $NAME $LR

echo "All Feature Extractors are trained!"
