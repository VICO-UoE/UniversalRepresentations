# Learning Universal Representations for Multiple Dense Prediction Tasks.

## Dependencies
- Python 3.6+
- PyTorch 1.0 (or newer version)
- torchvision 0.2.2 (or newer version)
- progress
- matplotlib
- numpy

## Usage

### Prepare dataset

We use the preprocessed [`NYUv2` dataset](https://www.dropbox.com/sh/86nssgwm6hm3vkb/AACrnUQ4GxpdrBbLjb6n-mWNa?dl=0) provided by [MTAN](https://github.com/lorenmt/mtan). Download the dataset and place the dataset folder in `./data/`

### Our method
#### Train Single Task Learning Network with SegNet:
```
CUDA_VISIBLE_DEVICES=<gpu-id> python nyu_stl.py --method single-task --backbone segnet --head segnet_head --task <task, e.g. semantic, depth, normal> --out ./results/stl --dataroot ./data/nyuv2
```
#### Train the Universal Representation Network:
```
CUDA_VISIBLE_DEVICES=<gpu-id> python nyu_url.py --backbone segnet --head segnet_head --tasks semantic depth normal --out ./results/url --single-dir ./results/stl --dataroot ./data/nyuv2

```

### Baselines
We also provide code `nyu_mtl_baseline.py` for several multi-task learning optimization strategies, including uniformly weighting [`uniform`]; [MGDA](https://arxiv.org/abs/1810.04650) [`mgda`], adapted from the [source code](https://github.com/intel-isl/MultiObjectiveOptimization); [GradNorm](https://arxiv.org/abs/1711.02257) [`gradnorm`]; [DWA](https://arxiv.org/abs/1803.10704) [`dwa`], from the [source code](https://github.com/lorenmt/mtan), [Uncertainty Weighting](https://openaccess.thecvf.com/content_cvpr_2018/papers/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.pdf) [`uncert`]; [Gradient Surgery](https://arxiv.org/abs/2001.06782) [`gs`] from [Pytorch-PCGrad](https://github.com/WeiChengTseng/Pytorch-PCGrad); [IMTL](https://openreview.net/forum?id=IMPnRXEWpvr) [`imtl_l`, `imtl_g`, `imtl_h`]; [GradDrop](https://arxiv.org/abs/2010.06808) [`gd`]; and [CAGrad](https://arxiv.org/abs/2110.14048) [`ca`] from the [source code](https://github.com/Cranial-XIX/CAGrad).

#### Train a Multi-task Learning Network using MGDA with SegNet:
```
CUDA_VISIBLE_DEVICES=<gpu-id> python nyu_mtl_baseline.py --backbone segnet --head segnet_head --tasks semantic depth normal --out ./results/mtl-baselines --dataroot ./data/nyuv2 --weight #weight: uniform, mgda, gradnorm, dwa, uncert, gs, imtl_l, imtl_g, imtl_h, gd, ca
```

## Model Zoo
[STL Models](https://drive.google.com/drive/folders/1EAYEjsUmc65x24P7kVwjZ51w_5AQCBZf) | [URL Model](https://drive.google.com/drive/folders/1jO85idM4EwyRL8BpNpGonTqpiYHxf53L)

## Acknowledge
We thank authors of [Multi-Task-Learning-PyTorch](https://github.com/SimonVandenhende/Multi-Task-Learning-PyTorch), [MTAN](https://github.com/lorenmt/mtan), [MGDA](https://github.com/intel-isl/MultiObjectiveOptimization), [Pytorch-PCGrad](https://github.com/WeiChengTseng/Pytorch-PCGrad), [CAGrad](https://github.com/Cranial-XIX/CAGrad) for their code!

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
}


