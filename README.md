# Universal Representations
We propose a <b>Universal Representation Learning</b> framework in (a) that generalizes over multi-task dense prediction tasks (b), multi-domain many-shot learning (c), cross-domain few-shot learning (d) by distilling knowledge of multiple task/domain-specific networks into a single deep neural network after aligning its representations with the task/domain-specific ones through small capacity adapters.

<p align="center">
  <img src="./figures/apps.png" style="width:100%">
  Figure 1. <b>Universal Representation Learning.</b>
</p>

> [**Universal Representations: A Unified Look at Multiple Task and Domain Learning**](https://arxiv.org/abs/2204.02744),            
> Wei-Hong Li, Xialei Liu, Hakan Bilen,        
> *Preprint 2022 ([arXiv 2204.02744](https://arxiv.org/abs/2204.02744))* 
>
> [**Universal Representation Learning from Multiple Domains for Few-shot Classification**](https://arxiv.org/abs/2103.13841),            
> Wei-Hong Li, Xialei Liu, Hakan Bilen,        
> *ICCV 2021 ([arXiv 2103.13841](https://arxiv.org/abs/2103.13841))*  
>
> [**Knowledge distillation for multi-task learning**](https://arxiv.org/abs/2007.06889),            
> Wei-Hong Li, Hakan Bilen,        
> *ECCV Workshop 2020 ([arXiv 2007.06889](https://arxiv.org/abs/2007.06889))*  

## Updates
* April'22, The preprint of our paper is now available! Code will be available soon! One can refer to [URL](https://github.com/VICO-UoE/URL) for the implementation on Cross-domain Few-shot Learning.

## Features at a glance
- We propose a unified look at jointly learning multiple vision tasks and visual domains through universal representations, a single deep neural network.

- We propose distilling knowledge of multiple task/domain-specific networks into a single deep neural network after aligning its representations with the task/domain-specific ones through small capacity adapters.

- We rigorously show that universal representations achieve state-of-the-art performances in learning of multiple dense prediction problems in NYU-v2 and Cityscapes, multiple image classification problems from diverse domains in Visual Decathlon Dataset and cross-domain few-shot learning in MetaDataset.

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
```