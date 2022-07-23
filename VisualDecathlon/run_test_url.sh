# Get results on the test set for evaluation (URL with ResNet-26)
CUDA_VISIBLE_DEVICES=0 python test_url.py --dataset imagenet12 aircraft cifar100 daimlerpedcls dtd gtsrb vgg-flowers omniglot svhn ucf101 --mode bn --expdir ./results/url_train+val/ --datadir ./data/decathlon-1.0/ --imdbdir ./data/decathlon-1.0/annotations/ --source ./results/url_train+val/checkpoint/ckptbnimagenet12aircraftcifar100daimlerpedclsdtdgtsrbvgg-flowersomniglotsvhnucf101_best.pth


# Get results on the test set for evaluation (URL with ResNet-26 + Parallel Adapters)
# CUDA_VISIBLE_DEVICES=0 python test_url.py --dataset imagenet12 aircraft cifar100 daimlerpedcls dtd gtsrb vgg-flowers omniglot svhn ucf101 --mode parallel_adapters --expdir ./results/url_ad_train+val/ --datadir ./data/decathlon-1.0/ --imdbdir ./data/decathlon-1.0/annotations/ --source ./results/url_ad_train+val/checkpoint/ckptparallel_adaptersimagenet12aircraftcifar100daimlerpedclsdtdgtsrbvgg-flowersomniglotsvhnucf101_best.pth
