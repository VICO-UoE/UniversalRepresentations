#
# Authors: Wei-Hong Li

import torch
import torch.nn as nn
import torch.nn.functional as F

def get_outfns(tasks):
    fns = {}
    for task in tasks:
        fns[task] = outfn(task)
    return fns

def outfn(task):
    if task == 'semantic':
        def fn(x):
            return F.log_softmax(x, dim=1)
    if task == 'depth':
        def fn(x):
            return x
    if task == 'normal':
        def fn(x):
            return F.normalize(x, p=2, dim=1)
    return fn

