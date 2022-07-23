# Adapted from https://github.com/SimonVandenhende/Multi-Task-Learning-PyTorch

import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
from models.out_fns import get_outfns

class SingleTaskModel(nn.Module):
    """ Single-task baseline model with encoder + decoder """
    def __init__(self, backbone: nn.Module, decoder: nn.Module, task: str):
        super(SingleTaskModel, self).__init__()
        self.backbone = backbone
        self.decoder = decoder 
        self.task = task
        self.outfns = get_outfns([task])

    def forward(self, x):
        out_size = x.size()[2:]
        feats = self.backbone(x)
        if isinstance(feats, list):
            feats = feats[-1]
        if feats.size()[2:] == out_size:
            out = self.decoder(feats)
        else:
            out = F.interpolate(self.decoder(feats), out_size, mode='bilinear', align_corners=True)
        return {self.task: self.outfns[self.task](out)}
    def embed(self, x):
        out_size = x.size()[2:]
        return self.backbone(x)


class MultiTaskModel(nn.Module):
    """ Multi-task baseline model with shared encoder + task-specific decoders """
    def __init__(self, backbone: nn.Module, decoders: nn.ModuleDict, tasks: list):
        super(MultiTaskModel, self).__init__()
        assert(set(decoders.keys()) == set(tasks))
        self.backbone = backbone
        self.decoders = decoders
        self.tasks = tasks
        self.outfns = get_outfns(tasks)

    def forward(self, x, feat=False):
        out_size = x.size()[2:]
        shared_representation = self.backbone(x)
        feats = shared_representation
        if isinstance(shared_representation, list):
            feats = shared_representation
            shared_representation = shared_representation[-1]
        if shared_representation.size()[2:] == out_size:
            if feat:
                return {task: self.outfns[task](self.decoders[task](shared_representation)) for task in self.tasks}, feats
            return {task: self.outfns[task](self.decoders[task](shared_representation)) for task in self.tasks}
        else:
            if feat:
                return {task: self.outfns[task](F.interpolate(self.decoders[task](shared_representation), out_size, mode='bilinear', align_corners=True)) for task in self.tasks}, feats
            return {task: self.outfns[task](F.interpolate(self.decoders[task](shared_representation), out_size, mode='bilinear', align_corners=True)) for task in self.tasks}
