# Learning Universal Representations over Diverse Visual Domains (Visual Decathlon Dataset).

## Dependencies
- Python 3.6+
- PyTorch 1.0 (or newer version)
- torchvision 0.2.2 (or newer version)
- kornia
- tqdm
- pycocotools
- numpy

## Usage

### Prepare dataset
Download the data with ``download_data.sh ./data``. You will need to download the imagenet dataset from https://image-net.org/download-images and put it in ./data/decathlon-1.0/data/.


### Our method
#### Train Single Domain Learning Network with ResNet-26 Backbone:
```
./run_train_sdl.sh
```
#### Train the Universal Representation Network with ResNet-26 Backbone:
```
./run_train_url.sh
```
#### Train the Universal Representation Network with ResNet-26 + Parallel Adapters as Backbone:
```
./run_train_url_ad.sh
```

#### Testing the Universal Representation Network:
Once the training of our URL method is done, one can test the results on the test split of each domain by the official online evaluation. Run the following script and submit the results to the [online evaluation website](https://competitions.codalab.org/competitions/17001).
```
./run_test_url.sh
```

Note that you will need to switch to the ResNet-26 + Parallel Adapters setting by setting `--mode` to `parallel_adapters`. See `./run_test_url.sh` for more details.

### Vanilla Multi-domain Learning Network
#### Train a vanilla Multi-domain Learning Network with ResNet-26:
```
./run_train_mdl.sh
```

## Model Zoo
[SDL Models](https://drive.google.com/drive/folders/1m4A6bUPd_s9F1qZXzd-VQ1rkciyXpsyf) | [SDL Models (Train+Val)](https://drive.google.com/drive/folders/1SIVX_Akli6TIdHWhZzStCu2aKDaudrOI) | [URL Model (Train+Val)](https://drive.google.com/drive/folders/19DTgubH9CjogQrSWCMnjrXPnUZcD2Set) | [URL (Parallel Adapter) Model (Train+Val)](https://drive.google.com/drive/folders/1891Zg45dzLPaOLL23YmBSzQdLt4RL9XK)

## Acknowledge
We thank authors of [residual_adapters](https://github.com/srebuffi/residual_adapters) for their source code!

## Contact
For any question, you can contact [Wei-Hong Li](https://weihonglee.github.io).

## Citation
If you use this code, please cite our papers:
```
@article{li2022Universal,
    author    = {Li, Wei-Hong and Liu, Xialei and Bilen, Hakan},
    title     = {Universal Representations: A Unified Look at Multiple Task and Domain Learning},
    journal   = {arXiv preprint arXiv:2204.02744},
    year      = {2022}
}

@inproceedings{li2021Universal,
    author    = {Li, Wei-Hong and Liu, Xialei and Bilen, Hakan},
    title     = {Universal Representation Learning From Multiple Domains for Few-Shot Classification},
    booktitle = {IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {9526-9535}
}

@inproceedings{li2020knowledge,
    author    = {Li, Wei-Hong and Bilen, Hakan},
    title     = {Knowledge distillation for multi-task learning},
    booktitle = {European Conference on Computer Vision (ECCV) Workshop},
    year      = {2020},
    xcode     = {https://github.com/VICO-UoE/KD4MTL}
} 

