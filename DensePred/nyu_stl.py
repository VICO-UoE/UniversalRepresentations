import os
import torch
import fnmatch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argparse
import torch.utils.data.sampler as sampler
import shutil
from dataset.nyuv2 import *
from torch.autograd import Variable
from models.get_model import get_model
from utils.evaluation import ConfMatrix, DepthMeter, NormalsMeter
from utils.dense_losses import get_dense_tasks_losses, get_task_loss, compute_miou, compute_iou, depth_error, normal_error
import numpy as np
from progress.bar import Bar as Bar
from utils import Logger, AverageMeter, accuracy, mkdir_p, savefig

parser = argparse.ArgumentParser(description='Single Task Learning (SegNet)')
parser.add_argument('--dataroot', default='nyuv2', type=str, help='dataset root')
parser.add_argument('--backbone', default='segnet', type=str, help='shared backbone')
parser.add_argument('--head', default='segnet_head', type=str, help='task-specific decoder')
parser.add_argument('--pretrained', dest='pretrained', action='store_true', help='using pretrained weight from ImageNet')
parser.add_argument('--dilated', dest='dilated', action='store_true', help='Dilated')
parser.add_argument('--fuse_hrnet', dest='fuse_hrnet', action='store_true', help='fuse_hrnet')
parser.add_argument('--method', default='single-task', type=str, help='vanilla or mtan')
parser.add_argument('--out', default='./results/stl', help='Directory to output the result')
parser.add_argument('--task', default='semantic', type=str, help='task: semantic, depth, normal')
opt = parser.parse_args()

tasks = ['semantic', 'depth', 'normal']

tasks_outputs = {
    'semantic': 13,
    'depth': 1,
    'normal': 3,
}

def save_checkpoint(state, is_best, checkpoint=opt.out, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, '{}_{}_stl_{}_'.format(opt.backbone, opt.head, opt.task) + filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, '{}_{}_stl_{}_'.format(opt.backbone, opt.head, opt.task) + 'model_best.pth.tar'))

if not os.path.isdir(opt.out):
    mkdir_p(opt.out)
title = 'NYUv2'
logger = Logger(os.path.join(opt.out, '{}_{}_stl_{}_log.txt'.format(opt.backbone, opt.head, opt.task)), title=title)
logger.set_names(['Epoch', 'T.Ls', 'T. mIoU', 'T. Pix', 'T.Ld', 'T.abs', 'T.rel', 'T.Ln', 'T.Mean', 'T.Med', 'T.11', 'T.22', 'T.30',
    'V.Ls', 'V. mIoU', 'V. Pix', 'V.Ld', 'V.abs', 'V.rel', 'V.Ln', 'V.Mean', 'V.Med', 'V.11', 'V.22', 'V.30'])

# define model, optimiser and scheduler
model = get_model(opt, tasks_outputs=tasks_outputs).cuda()

optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

if opt.task == 'semantic':
    print('LOSS FORMAT: SEMANTIC_LOSS MEAN_IOU PIX_ACC\n')
elif opt.task == 'depth':
    print('LOSS FORMAT: DEPTH_LOSS ABS_ERR REL_ERR\n')
elif opt.task == 'normal':
    print('LOSS FORMAT: NORMAL_LOSS MEAN MED <11.25 <22.5 <30\n')

# define dataset path
dataset_path = opt.dataroot
nyuv2_train_set = NYUv2(root=dataset_path, train=True, augmentation=True, flip=True, normalize=False)
nyuv2_test_set = NYUv2(root=dataset_path, train=False, normalize=False)

batch_size = 2
nyuv2_train_loader = torch.utils.data.DataLoader(
    dataset=nyuv2_train_set,
    batch_size=batch_size,
    shuffle=True, num_workers=0, drop_last=False)


nyuv2_test_loader = torch.utils.data.DataLoader(
    dataset=nyuv2_test_set,
    batch_size=batch_size,
    shuffle=True, num_workers=0)


# define parameters
total_epoch = 200
train_batch = len(nyuv2_train_loader)
test_batch = len(nyuv2_test_loader)
avg_cost = np.zeros([total_epoch, 24], dtype=np.float32)
best_performance = -100
isbest=False
for epoch in range(total_epoch):
    index = epoch
    cost = np.zeros(24, dtype=np.float32)
    scheduler.step()

    bar = Bar('Training', max=train_batch)

    # iteration for all batches
    model.train()
    nyuv2_train_dataset = iter(nyuv2_train_loader)
    for k in range(train_batch):
        train_data, train_label, train_depth, train_normal = nyuv2_train_dataset.next()
        train_data, train_label = train_data.cuda(), train_label.type(torch.LongTensor).cuda()
        train_depth, train_normal = train_depth.cuda(), train_normal.cuda()
        train_labels = {'semantic': train_label, 'depth': train_depth, 'normal': train_normal}

        train_pred = model(train_data)
        train_loss = get_task_loss(train_pred[opt.task], train_labels[opt.task], task=opt.task)
        loss = train_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if opt.task == 'semantic':
            cost[0] = train_loss.item()
            cost[1] = compute_miou(train_pred['semantic'], train_label, tasks_outputs['semantic']).item()
            cost[2] = compute_iou(train_pred['semantic'], train_label).item()
        elif opt.task == 'depth':
            cost[3] = train_loss.item()
            cost[4], cost[5] = depth_error(train_pred['depth'], train_depth)
        elif opt.task == 'normal':
            cost[6] = train_loss.item()
            cost[7], cost[8], cost[9], cost[10], cost[11] = normal_error(train_pred['normal'], train_normal)
        avg_cost[index, :12] += cost[:12] / train_batch
        if opt.task =='semantic':
            bar.suffix  = '({batch}/{size}) | LossS: {loss_s:.4f}'.format(
                    batch=k + 1,
                    size=train_batch,
                    loss_s=avg_cost[index,1],
                    )
        elif opt.task == 'depth':
            bar.suffix  = '({batch}/{size}) | LossD: {loss_d:.4f}'.format(
                    batch=k + 1,
                    size=train_batch,
                    loss_d=avg_cost[index,3],
                    )
        elif opt.task == 'normal':
            bar.suffix  = '({batch}/{size}) | LossN: {loss_n:.4f}'.format(
                    batch=k + 1,
                    size=train_batch,
                    loss_n=avg_cost[index,6],
                    )
        bar.next()
    bar.finish()

    # evaluating test data
    model.eval()
    conf_mat = ConfMatrix(tasks_outputs['semantic'])
    depth_mat = DepthMeter()
    normal_mat = NormalsMeter()
    with torch.no_grad():  # operations inside don't track history
        nyuv2_test_dataset = iter(nyuv2_test_loader)
        for k in range(test_batch):
            test_data, test_label, test_depth, test_normal = nyuv2_test_dataset.next()
            test_data, test_label = test_data.cuda(),  test_label.type(torch.LongTensor).cuda()
            test_depth, test_normal = test_depth.cuda(), test_normal.cuda()

            test_labels = {'semantic': test_label, 'depth': test_depth, 'normal': test_normal}

            test_pred = model(test_data)

            test_loss = get_task_loss(test_pred[opt.task], test_labels[opt.task], task=opt.task)

            if opt.task == 'semantic':
                conf_mat.update(test_pred['semantic'].argmax(1).flatten(), test_label.flatten())
                cost[12] = test_loss.item()
            elif opt.task == 'depth':
                depth_mat.update(test_pred['depth'], test_depth)
                cost[15] = test_loss.item()
            elif opt.task == 'normal':
                normal_mat.update(test_pred['normal'], test_normal)
                cost[18] = test_loss.item()
            
            avg_cost[index, 12:] += cost[12:] / test_batch
        if opt.task == 'semantic':
            avg_cost[index, 13:15] = conf_mat.get_metrics()
        elif opt.task == 'depth':
            depth_metric = depth_mat.get_score()
            avg_cost[index, 16], avg_cost[index, 17] = depth_metric['l1'], depth_metric['rmse']
        elif opt.task == 'normal':
            normal_metric = normal_mat.get_score()
            avg_cost[index, 19], avg_cost[index, 20], avg_cost[index, 21], avg_cost[index, 22], avg_cost[index, 23] = normal_metric['mean'], normal_metric['rmse'], normal_metric['11.25'], normal_metric['22.5'], normal_metric['30']

    if opt.task == 'semantic':
        avg_cost[index, 13:15] = conf_mat.get_metrics()
        stl_performance = avg_cost[index, 13]
    if opt.task == 'depth':
        depth_metric = depth_mat.get_score()
        avg_cost[index, 16], avg_cost[index, 17] = depth_metric['l1'], depth_metric['rmse']
        stl_performance = - avg_cost[index, 16]
    if opt.task == 'normal':
        normal_metric = normal_mat.get_score()
        avg_cost[index, 19], avg_cost[index, 20], avg_cost[index, 21], avg_cost[index, 22], avg_cost[index, 23] = normal_metric['mean'], normal_metric['rmse'], normal_metric['11.25'], normal_metric['22.5'], normal_metric['30']
        stl_performance = - avg_cost[index, 19]

    isbest = stl_performance > best_performance



    if opt.task == 'semantic':
        print('Epoch: {:04d} | TRAIN: {:.4f} {:.4f} {:.4f}'
          'TEST: {:.4f} {:.4f} {:.4f}'
          .format(index, avg_cost[index, 0], avg_cost[index, 1], avg_cost[index, 2], 
                avg_cost[index, 12], avg_cost[index, 13], avg_cost[index, 14]))
    elif opt.task == 'depth':
        print('Epoch: {:04d} | TRAIN: {:.4f} {:.4f} {:.4f}'
          'TEST: {:.4f} {:.4f} {:.4f}'
          .format(index, avg_cost[index, 3], avg_cost[index, 4], avg_cost[index, 5], 
                avg_cost[index, 15], avg_cost[index, 16], avg_cost[index, 17]))
    elif opt.task == 'normal':
        print('Epoch: {:04d} | TRAIN: {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} '
          'TEST: {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}'
          .format(index, avg_cost[index, 6], avg_cost[index, 7], avg_cost[index, 8], avg_cost[index, 9], avg_cost[index, 10], avg_cost[index, 11],
                avg_cost[index, 18], avg_cost[index, 19], avg_cost[index, 20], avg_cost[index, 21], avg_cost[index, 22], avg_cost[index, 23]))


    logger.append([index, avg_cost[index, 0], avg_cost[index, 1], avg_cost[index, 2], avg_cost[index, 3],
                avg_cost[index, 4], avg_cost[index, 5], avg_cost[index, 6], avg_cost[index, 7], avg_cost[index, 8], avg_cost[index, 9],
                avg_cost[index, 10], avg_cost[index, 11], avg_cost[index, 12], avg_cost[index, 13],
                avg_cost[index, 14], avg_cost[index, 15], avg_cost[index, 16], avg_cost[index, 17], avg_cost[index, 18],
                avg_cost[index, 19], avg_cost[index, 20], avg_cost[index, 21], avg_cost[index, 22], avg_cost[index, 23]])

    if isbest:
        best_performance = stl_performance
        best_index = index
    save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_performance': best_performance,
            'optimizer' : optimizer.state_dict(),
        }, isbest) 

if opt.task == 'semantic':
    print('Epoch: {:04d} | TRAIN: {:.4f} {:.4f} {:.4f}'
        'TEST: {:.4f} {:.4f} {:.4f}'
        .format(best_index, avg_cost[best_index, 0], avg_cost[best_index, 1], avg_cost[best_index, 2], 
            avg_cost[best_index, 12], avg_cost[best_index, 13], avg_cost[best_index, 14]))
elif opt.task == 'depth':
    print('Epoch: {:04d} | TRAIN: {:.4f} {:.4f} {:.4f}'
        'TEST: {:.4f} {:.4f} {:.4f}'
        .format(best_index, avg_cost[best_index, 3], avg_cost[best_index, 4], avg_cost[best_index, 5], 
            avg_cost[best_index, 15], avg_cost[best_index, 16], avg_cost[best_index, 17]))
elif opt.task == 'normal':
    print('Epoch: {:04d} | TRAIN: {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} '
        'TEST: {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}'
        .format(best_index, avg_cost[best_index, 6], avg_cost[best_index, 7], avg_cost[best_index, 8], avg_cost[best_index, 9], avg_cost[best_index, 10], avg_cost[best_index, 11],
            avg_cost[best_index, 18], avg_cost[best_index, 19], avg_cost[best_index, 20], avg_cost[best_index, 21], avg_cost[best_index, 22], avg_cost[best_index, 23]))
