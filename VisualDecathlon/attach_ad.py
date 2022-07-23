# Authors: Wei-Hong Li
# Attaching adapters

import torch.nn as nn
import config_task
import copy
import pdb

def conv1x1_fonc(in_planes, out_planes=None, stride=1, bias=False):
    if out_planes is None:
        return nn.Conv2d(in_planes, in_planes, kernel_size=1, stride=stride, padding=0, bias=bias)
    else:
        return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=bias)

class conv1x1(nn.Module):
    
    def __init__(self, planes, out_planes=None, stride=1):
        super(conv1x1, self).__init__()
        if config_task.mode == 'series_adapters':
            self.conv = nn.Sequential(nn.BatchNorm2d(planes), conv1x1_fonc(planes))
        elif config_task.mode == 'parallel_adapters':
            self.conv = conv1x1_fonc(planes, out_planes, stride) 
        else:
            self.conv = conv1x1_fonc(planes)
    def forward(self, x):
        y = self.conv(x)
        if config_task.mode == 'series_adapters':
            y += x
        return y

class conv_ad(nn.Module):
    def __init__(self, orig_conv):
        super(conv_ad, self).__init__()
        self.conv = copy.deepcopy(orig_conv)
        # self.conv.weight.requires_grad = False
        planes, in_planes, _, _ = self.conv.weight.size()
        stride, _ = self.conv.stride
        # pdb.set_trace()
        if config_task.mode == 'parallel_adapters':
            self.parallel_conv = nn.ModuleList([conv1x1(in_planes, planes, stride) for i in range(len(config_task.all_tasks))])
            for conv in self.parallel_conv:
                conv.requires_grad=True
    def forward(self, x):
        task = config_task.task
        # print(task)
        y = self.conv(x)
        if config_task.mode == 'parallel_adapters':
            y = y + self.parallel_conv[task](x)

        return y

class domain_bn(nn.Module):
    def __init__(self, bn):
        super(domain_bn, self).__init__()
        self.bns = nn.ModuleList([copy.deepcopy(bn) for i in range(len(config_task.all_tasks))])
        for bn in self.bns:
            bn.requires_grad=True
    def forward(self, x):
        task = config_task.task
        y = self.bns[task](x)

        return y

def attach_ads(orig_resnet):
    orig_resnet.conv1 = conv_ad(orig_resnet.conv1)
    orig_resnet.bn1 = domain_bn(orig_resnet.bn1)
    for block in orig_resnet.layer1:
        for name, m in block.named_children():
            if isinstance(m, nn.Conv2d) and m.kernel_size[0] == 3:
                new_conv = conv_ad(m)
                setattr(block, name, new_conv)
            if isinstance(m, nn.BatchNorm2d):
                new_bn = domain_bn(m)
                setattr(block, name, new_bn)

    for block in orig_resnet.layer2:
        for name, m in block.named_children():
            if isinstance(m, nn.Conv2d) and m.kernel_size[0] == 3:
                new_conv = conv_ad(m)
                setattr(block, name, new_conv)
            if isinstance(m, nn.BatchNorm2d):
                new_bn = domain_bn(m)
                setattr(block, name, new_bn)

    for block in orig_resnet.layer3:
        for name, m in block.named_children():
            if isinstance(m, nn.Conv2d) and m.kernel_size[0] == 3:
                new_conv = conv_ad(m)
                setattr(block, name, new_conv)
            if isinstance(m, nn.BatchNorm2d):
                new_bn = domain_bn(m)
                setattr(block, name, new_bn)

    orig_resnet.end_bn = domain_bn(orig_resnet.end_bn)
    return orig_resnet

