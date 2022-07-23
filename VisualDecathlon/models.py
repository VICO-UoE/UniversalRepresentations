#
# Authors: Wei-Hong Li
# This code is adapted from https://github.com/srebuffi/residual_adapters
# We use the default ResNet-26 as in https://github.com/srebuffi/residual_adapters 
# and remove the adapters implementation.

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torch.nn.parameter import Parameter
import config_task
import math


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

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

class conv_task(nn.Module):
    
    def __init__(self, in_planes, planes, stride=1, nb_tasks=1, is_proj=1, second=0):
        super(conv_task, self).__init__()
        self.is_proj = is_proj
        self.second = second
        self.conv = conv3x3(in_planes, planes, stride)
        if config_task.mode == 'series_adapters' and is_proj:
            self.bns = nn.ModuleList([nn.Sequential(conv1x1(planes), nn.BatchNorm2d(planes)) for i in range(nb_tasks)])
        elif config_task.mode == 'parallel_adapters' and is_proj:
            self.parallel_conv = nn.ModuleList([conv1x1(in_planes, planes, stride) for i in range(nb_tasks)])
            self.bns = nn.ModuleList([nn.BatchNorm2d(planes) for i in range(nb_tasks)])
        else:
            self.bns = nn.ModuleList([nn.BatchNorm2d(planes) for i in range(nb_tasks)])
    
    def forward(self, x):
        task = config_task.task
        y = self.conv(x)
        if self.second == 0:
            if config_task.isdropout1:
                x = F.dropout2d(x, p=0.5, training = self.training)
        else:
            if config_task.isdropout2:
                x = F.dropout2d(x, p=0.5, training = self.training)
        if config_task.mode == 'parallel_adapters' and self.is_proj:
            y = y + self.parallel_conv[task](x)
        y = self.bns[task](y)

        return y

# No projection: identity shortcut
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, shortcut=0):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(True)
        self.conv2 = conv3x3(planes, planes, 1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = shortcut
        if self.shortcut == 1:
            self.avgpool = nn.AvgPool2d(2)
        
    def forward(self, x):
        residual = x
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.bn2(y)
        if self.shortcut == 1:
            residual = self.avgpool(x)
            residual = torch.cat((residual, residual*0),1)
        y += residual
        y = F.relu(y)
        return y


class ResNet(nn.Module):
    def __init__(self, block, nblocks, num_classes=[10]):
        super(ResNet, self).__init__()
        self.num_classes = num_classes
        blocks = [block, block, block]
        factor = config_task.factor
        self.in_planes = int(32*factor)
        self.conv1 = conv3x3(3, int(32*factor), 1)
        self.bn1 = nn.BatchNorm2d(int(32*factor))
        self.layer1 = self._make_layer(blocks[0], int(64*factor), nblocks[0], stride=2)
        self.layer2 = self._make_layer(blocks[1], int(128*factor), nblocks[1], stride=2)
        self.layer3 = self._make_layer(blocks[2], int(256*factor), nblocks[2], stride=2)
        self.end_bn = nn.BatchNorm2d(int(256*factor))
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        if self.num_classes is not None:
            if isinstance(num_classes, list):
                if len(num_classes) == 1:
                    self.cls_fn = nn.Linear(256 * block.expansion, num_classes[0])
                else:
                    cls_fn = []
                    for num_c in num_classes:
                        cls_fn.append(nn.Linear(256 * block.expansion, num_c))
                    self.cls_fn = nn.ModuleList(cls_fn)
            else:
                self.cls_fn = nn.Linear(256 * block.expansion, num_classes)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    
    def _make_layer(self, block, planes, nblocks, stride=1):
        shortcut = 0
        if stride != 1 or self.in_planes != planes * block.expansion:
            shortcut = 1
        layers = []
        layers.append(block(self.in_planes, planes, stride, shortcut))
        self.in_planes = planes * block.expansion
        for i in range(1, nblocks):
            layers.append(block(self.in_planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x, kd=False, num_samples=None, task=None):
        feat = self.embed(x)
        if self.num_classes is not None:
            if task is not None:
                x = self.cls_fn[task](feat)
            else:
                if isinstance(self.num_classes, list) and len(self.num_classes)>1:
                    if num_samples is not None:
                        feat = torch.split(feat, num_samples)
                    else:
                        feat = torch.split(feat, int(x.size(0)/len(self.num_classes)))
                    x = []
                    for t in range(len(self.num_classes)):
                        x.append(self.cls_fn[t](feat[t]))
                else:
                    x = self.cls_fn(feat)
        # else:
            # x = self.cls_fn(feat)
        if kd:
            return x, feat
        else:
            return x

    def embed(self, x):
        x = self.conv1(x)
        x = self.bn1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.end_bn(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x




def resnet26(num_classes=10, blocks=BasicBlock):
    return  ResNet(blocks, [4,4,4],num_classes)


