# Train the URL model with ResNet-26
CUDA_VISIBLE_DEVICES=0 python train_url.py --dataset imagenet12 aircraft cifar100 daimlerpedcls dtd gtsrb vgg-flowers omniglot svhn ucf101 --trainval --wd 1. --lr 0.01 --alr 1e-2 --mode bn --expdir ./results/url_train+val/ --datadir ./data/decathlon-1.0/ --imdbdir ./data/decathlon-1.0/annotations/ --sdl-root ./results/sdl_train+val --adaptor-opt linear --sigma 1 --beta 1 --source ./results/sdl_train+val/checkpoint/ckptbnimagenet12_best.pth

