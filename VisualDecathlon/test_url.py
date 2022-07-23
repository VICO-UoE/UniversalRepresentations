#
# Authors: Wei-Hong Li

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

import models
import os
import time
import argparse
import numpy as np
import pickle

from torch.autograd import Variable
import datasets
import config_task
import utils_pytorch
from utils import adaptor, distillation_loss, WeightAnnealing

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json

parser = argparse.ArgumentParser(description='PyTorch Universal Representation Learning')
parser.add_argument('--dataset', default='cifar100', nargs='+', help='Task(s) to be trained')
parser.add_argument('--mode', default='parallel_adapters', type=str, help='Task adaptation mode')
parser.add_argument('--expdir', default='./results/url/', help='Save folder')
parser.add_argument('--datadir', default='./data/decathlon-1.0/', help='folder containing data folder')
parser.add_argument('--imdbdir', default='./data/decathlon-1.0/annotations/', help='annotation folder')
parser.add_argument('--source', default='', type=str, help='Network source')
parser.add_argument('--seed', default=0, type=int, help='seed')
parser.add_argument('--factor', default='1.', type=float, help='Width factor of the network')

args = parser.parse_args()
config_task.mode = args.mode
config_task.factor = args.factor
config_task.all_tasks = ['imagenet12', 'aircraft', 'cifar100', 'daimlerpedcls', 'dtd', 'gtsrb', 'vgg-flowers', 'omniglot', 'svhn', 'ucf101']
args.use_cuda = torch.cuda.is_available()
if type(args.dataset) is str:
    args.dataset = [args.dataset]

if not os.path.isdir(args.expdir):
    os.mkdir(args.expdir) 
print('=> results and checkpoints will be stored in {}'.format(args.expdir))

args.ckpdir = args.expdir + '/checkpoint/'
args.svdir  = args.expdir + '/results/'
args.testdir = args.expdir + '/test-results/'

if not os.path.isdir(args.ckpdir):
    os.mkdir(args.ckpdir) 

if not os.path.isdir(args.svdir):
    os.mkdir(args.svdir) 

if not os.path.isdir(args.testdir):
    os.mkdir(args.testdir)

#####################################

# Prepare data loaders
train_loaders, val_loaders, num_classes = datasets.prepare_data_loaders(args.dataset,args.datadir,args.imdbdir,True)
test_loaders = datasets.test_data_loaders(args.dataset,args.datadir,args.imdbdir)
args.num_classes = num_classes
num_classes = {}
for t_index, task in enumerate(args.dataset):
    num_classes[task] = args.num_classes[t_index]

# Create the network
net = models.resnet26(num_classes=list(num_classes.values()))
net.eval()
if config_task.mode == 'parallel_adapters':
    from attach_ad import attach_ads
    net = attach_ads(net)
if args.source:
    checkpoint = torch.load(args.source)
    state_dict = checkpoint['net']
    net.load_state_dict(state_dict, strict=False)


all_tasks = range(len(args.dataset))
np.random.seed(1993)

if args.use_cuda:
    net.cuda()
    cudnn.benchmark = True

# test the model on validation set
ans_val = utils_pytorch.urltestset(val_loaders, all_tasks, net, args, ad=(config_task.mode=='parallel_adapters'))
utils_pytorch.coco_results_val(ans_val, args.datadir, args.imdbdir, args, args.testdir)

# test the model on test set
ans = utils_pytorch.urltestset(test_loaders, all_tasks, net, args, ad=(config_task.mode=='parallel_adapters'))
utils_pytorch.coco_results(ans, args.datadir, args.imdbdir, args, args.testdir)



