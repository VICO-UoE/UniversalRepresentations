# Train Vanilla Multi-domain Learning Network with ResNet-26
CUDA_VISIBLE_DEVICES=0 python train_mdl.py --dataset imagenet12 aircraft cifar100 daimlerpedcls dtd gtsrb vgg-flowers omniglot svhn ucf101 --wd 1. --lr 0.01 --mode bn --expdir ./results/mdl/ --datadir ./data/decathlon-1.0/ --imdbdir ./data/decathlon-1.0/annotations/ --source ./results/sdl/checkpoint/ckptbnimagenet12_best.pth 
