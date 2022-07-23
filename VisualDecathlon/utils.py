import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
import models
from cka import linear_CKA, kernel_CKA
from torch.optim.lr_scheduler import (MultiStepLR, ExponentialLR,
                                      CosineAnnealingWarmRestarts,
                                      CosineAnnealingLR)
import pdb


def create_model(args, num_classes, resume=None):
    model = models.resnet26(num_classes=num_classes)
    model.eval()

    if resume is not None:
        checkpoint = torch.load(resume)
        model.load_state_dict(checkpoint['net'], strict=False)
    if args.use_cuda:
        model.cuda()
    return model


def get_multinet_extractor(trainsets, args, MODEL_NAME, num_classes=None, SR=False):
    extractors = dict()
    for dataset_name in trainsets:
        if num_classes is not None:
            num_classes_ = num_classes[dataset_name]
        resume = os.path.join(args.sdl_root, 'checkpoint', 'ckptbn{}.pth'.format(dataset_name))
        if not os.path.isfile(resume):
            resume = os.path.join(args.sdl_root, 'checkpoint', 'ckptbn{}_best.pth'.format(dataset_name))
        print('==> loading {}'.format(resume))
        if num_classes_ is None:
            extractor = create_model(args=args, num_classes=None, resume=resume)
        else:
            # extractor = get_model(num_classes[dataset_name], args)
            extractor = create_model(args=args, num_classes=num_classes_, resume=resume)
        extractor.eval()
        extractors[dataset_name] = extractor

    def embed_many(images, return_type='dict', kd=False, logits=False, kd_layers=False):
        with torch.no_grad():
            all_features = dict()
            all_logits = dict()
            for name, extractor in extractors.items():
                if logits:
                    if kd:
                        all_logits[name], all_features[name] = extractor(images[name], kd=True)
                    elif kd_layers:
                        all_logits[name], all_features[name] = extractor.forward_layers(images[name], kd=True)
                    else:
                        all_logits[name] = extractor(images[name])
                else:
                    if kd:
                        all_features[name] = extractor.embed(images[name])
                    else:
                        all_features[name] = extractor.embed(images)

        if return_type == 'list':
            return list(all_features.values()), list(all_logits.values())
        else:
            return all_features, all_logits
    def logits_many(features, return_type='dict'):
        all_logits = dict()
        for t, (name, extractor) in enumerate(extractors.items()):
            all_logits[name] = extractor.cls_fn(features[t])
        if return_type == 'list':
            return list(all_logits.values())
        else:
            return all_logits
    if SR:
        return embed_many, logits_many
    return embed_many

def cal_dist(inputs, inputs_center):
    n = inputs.size(0)
    # Compute pairwise distance, replace by the official when merged
    dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
    dist = dist + dist.t()
    dist.addmm_(1, -2, inputs, inputs.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    # Compute pairwise distance, replace by the official when merged
    dist_center = torch.pow(inputs_center, 2).sum(dim=1, keepdim=True).expand(n, n)
    dist_center = dist_center + dist_center.t()
    dist_center.addmm_(1, -2, inputs_center, inputs_center.t())
    dist_center = dist_center.clamp(min=1e-12).sqrt()  # for numerical stability
    loss = torch.mean(torch.norm(dist-dist_center,p=2))
    return loss

def distillation_loss(fs, ft, opt='l2', delta=0.5):
    if opt == 'l2':
        return (fs-ft).pow(2).sum(1).mean()
    if opt == 'l1':
        return (fs-ft).abs().sum(1).mean()
    if opt == 'huber':
        l1 = (fs-ft).abs()
        binary_mask_l1 = (l1.sum(1) > delta).type(torch.FloatTensor).unsqueeze(1).cuda()
        binary_mask_l2 = (l1.sum(1) <= delta).type(torch.FloatTensor).unsqueeze(1).cuda()
        loss = (l1.pow(2) * binary_mask_l2 * 0.5).sum(1) + (l1 * binary_mask_l1).sum(1) * delta - delta ** 2 * 0.5
        loss = loss.mean()
        return loss
    if opt == 'rkd':
        return cal_dist(fs, ft)
    if opt == 'cosine':
        return 1 - F.cosine_similarity(fs, ft, dim=-1, eps=1e-30).mean()
    if opt == 'linearcka':
        return 1 - linear_CKA(fs, ft)
    if opt == 'kernelcka':
        return 1 - kernel_CKA(fs, ft)

class DistillKL(nn.Module):
    """KL divergence for distillation"""
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        loss = F.kl_div(p_s, p_t, reduction='sum') * (self.T**2) / y_s.shape[0]
        return loss

class adaptor(torch.nn.Module):
    def __init__(self, num_datasets, dim_in, dim_out=None, opt='linear'):
        super(adaptor, self).__init__()
        if dim_out is None:
            dim_out = dim_in
        self.num_datasets = num_datasets

        for i in range(num_datasets):
            if opt == 'linear':
                setattr(self, 'conv{}'.format(i), torch.nn.Conv2d(dim_in, dim_out, 1, bias=False))
            else:
                setattr(self, 'conv{}'.format(i), nn.Sequential(
                    torch.nn.Conv2d(dim_in, 2*dim_in, 1, bias=False),
                    torch.nn.ReLU(True),
                    torch.nn.Conv2d(2*dim_in, dim_out, 1, bias=False),
                    )
                )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)

            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
    def forward(self, inputs):
        results = []
        for i in range(self.num_datasets):
            ad_layer = getattr(self, 'conv{}'.format(i))
            if len(list(inputs[i].size())) < 4:
                input_ = inputs[i].view(inputs[i].size(0), -1, 1, 1)
            else:
                input_ = inputs[i]
            results.append(ad_layer(input_).flatten(1))
            # results.append(ad_layer(inputs[i]))
        return results
    def forward_task(self, inputs, task):
        ad_layer = getattr(self, 'conv{}'.format(task))
        if len(list(inputs.size())) < 4:
            input_ = inputs.view(inputs.size(0), -1, 1, 1)
        else:
            input_ = inputs
        results = ad_layer(input_).flatten(1)
        return results


class WeightAnnealing(nn.Module):
    """WeightAnnealing"""
    def __init__(self, T, alpha=10):
        super(WeightAnnealing, self).__init__()
        self.T = T
        self.alpha = alpha

    def forward(self, t, opt='exp'):
        if t > self.T:
            return 0
        if opt == 'exp':
            return 1-np.exp(self.alpha*((t)/self.T-1))
        if opt == 'log':
            return np.exp(-(t)/self.T*self.alpha)
        if opt == 'linear':
            return 1-(t)/self.T

class CosineAnnealRestartLR(object):
    def __init__(self, optimizer, args, start_iter):
        self.iter = start_iter
        self.max_iter = args.nb_epochs
        self.lr_scheduler = CosineAnnealingWarmRestarts(
            optimizer, 20, last_epoch=start_iter-1)

    def step(self, _iter):
        self.iter += 1
        self.lr_scheduler.step(_iter)
        stop_training = self.iter >= self.max_iter
        return stop_training


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



class CosineAnnealRestartLR(object):
    def __init__(self, optimizer, args, start_iter):
        self.iter = start_iter
        self.max_iter = args.nb_epochs
        self.lr_scheduler = CosineAnnealingWarmRestarts(
            optimizer, 20, last_epoch=start_iter-1)
        # self.lr_scheduler = CosineAnnealingLR(
        #     optimizer, args['train.cosine_anneal_freq'], last_epoch=start_iter-1)

    def step(self, _iter):
        self.iter += 1
        self.lr_scheduler.step(_iter)
        stop_training = self.iter >= self.max_iter
        return stop_training