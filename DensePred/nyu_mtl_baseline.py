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
import numpy as np
from progress.bar import Bar as Bar
from utils import Logger, AverageMeter, accuracy, mkdir_p, savefig
from utils.dense_losses import get_dense_tasks_losses, get_task_loss, compute_miou, compute_iou, depth_error, normal_error
from utils.pcgrad import PCGrad
from utils.graddrop import GDGrad
from utils.cagrad import CAGrad
from utils.imtl import IMTL
from torch.autograd import Variable
from mgda.min_norm_solvers import MinNormSolver, gradient_normalizers

parser = argparse.ArgumentParser(description='Baselines (SegNet)')
parser.add_argument('--weight', default='uniform', type=str, help='multi-task weighting: uniform, gradnorm, mgda, uncert, dwa, gs')
parser.add_argument('--backbone', default='resnet50', type=str, help='shared backbone')
parser.add_argument('--head', default='deeplab', type=str, help='task-specific decoder')
parser.add_argument('--tasks', default='semantic', nargs='+', help='Task(s) to be trained')
parser.add_argument('--pretrained', dest='pretrained', action='store_true', help='using pretrained weight from ImageNet')
parser.add_argument('--dilated', dest='dilated', action='store_true', help='Dilated')
parser.add_argument('--method', default='vanilla', type=str, help='vanilla or mtan')
parser.add_argument('--dataroot', default='nyuv2', type=str, help='dataset root')
parser.add_argument('--temp', default=1.0, type=float, help='temperature for DWA (must be positive)')
parser.add_argument('--fuse_hrnet', dest='fuse_hrnet', action='store_true', help='fuse_hrnet')
parser.add_argument('--wlr', default=0.001, type=float, help='initial learning rate')
parser.add_argument('--out', default='./results/mtl-baselines', help='Directory to output the result')
parser.add_argument('--alpha', default=1.5, type=float, help='hyper params of GradNorm')
opt = parser.parse_args()

tasks = ['semantic', 'depth', 'normal']

tasks_outputs = {
    'semantic': 13,
    'depth': 1,
    'normal': 3,
}

stl_performance = {}

stl_performance['segnet_segnet_head'] = {
                    'full': {'semantic': 40.5355, 'depth': 0.627602, 'normal': 24.284388},
}

class Weight(torch.nn.Module):
    def __init__(self, tasks):
        super(Weight, self).__init__()
        self.weights = nn.Parameter(torch.FloatTensor([1.0, 1.0, 1.0]))

def save_checkpoint(state, is_best, checkpoint=opt.out, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, '{}_{}_mtl_baselines_{}_{}_'.format(opt.backbone, opt.head, opt.method, opt.weight) + filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, '{}_{}_mtl_baselines_{}_{}_'.format(opt.backbone, opt.head, opt.method, opt.weight) + 'model_best.pth.tar'))

if not os.path.isdir(opt.out):
    mkdir_p(opt.out)
title = 'NYUv2'
logger = Logger(os.path.join(opt.out, '{}_{}_mtl_baselines_{}_{}_log.txt'.format(opt.backbone, opt.head, opt.method, opt.weight)), title=title)
logger.set_names(['Epoch', 'T.Ls', 'T. mIoU', 'T. Pix', 'T.Ld', 'T.abs', 'T.rel', 'T.Ln', 'T.Mean', 'T.Med', 'T.11', 'T.22', 'T.30',
    'V.Ls', 'V. mIoU', 'V. Pix', 'V.Ld', 'V.abs', 'V.rel', 'V.Ln', 'V.Mean', 'V.Med', 'V.11', 'V.22', 'V.30', 'Ws', 'Wd', 'Wn'])

# define model, optimiser and scheduler
model = get_model(opt, tasks_outputs=tasks_outputs).cuda()
Weights = Weight(tasks).cuda()
params = []
params += model.parameters()
params += [Weights.weights]
optimizer = optim.Adam(params, lr=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
if opt.weight == 'uncert':
    Weights.weights.data *= -0.5
if opt.weight == 'gs':
    optimizer = PCGrad(optimizer, reduction='mean')
if 'imtl' in opt.weight:
    Weights.weights.data *= 0
    optimizer = IMTL(optimizer, reduction='sum')
if opt.weight == 'gd':
    optimizer = GDGrad(optimizer)
if opt.weight == 'ca':
    optimizer = CAGrad(optimizer)


# compute parameter space
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print('Parameter Space: ABS: {:.1f}, REL: {:.4f}\n'.format(count_parameters(model),
                                                           count_parameters(model)/24981069))
print('LOSS FORMAT: SEMANTIC_LOSS MEAN_IOU PIX_ACC | DEPTH_LOSS ABS_ERR REL_ERR | NORMAL_LOSS MEAN MED <11.25 <22.5 <30\n')

# define dataset path
dataset_path = opt.dataroot
nyuv2_train_set = NYUv2(root=dataset_path, train=True, augmentation=True, flip=True, normalize=False)
nyuv2_test_set = NYUv2(root=dataset_path, train=False, normalize=False)

batch_size = 2
nyuv2_train_loader = torch.utils.data.DataLoader(
    dataset=nyuv2_train_set,
    batch_size=batch_size,
    shuffle=True, num_workers=0)

nyuv2_test_loader = torch.utils.data.DataLoader(
    dataset=nyuv2_test_set,
    batch_size=batch_size,
    shuffle=False, num_workers=0)


# define parameters
total_epoch = 200
train_batch = len(nyuv2_train_loader)
test_batch = len(nyuv2_test_loader)
T = opt.temp
avg_cost = np.zeros([total_epoch, 24], dtype=np.float32)
lambda_weight = np.zeros([3, total_epoch])
best_performance = -100
isbest = False
for epoch in range(total_epoch):
    index = epoch
    cost = np.zeros(24, dtype=np.float32)
    scheduler.step()

    # apply Dynamic Weight Average (DWA)
    if opt.weight == 'dwa':
        if index == 0 or index == 1:
            lambda_weight[:, index] = 1.0
        else:
            w_1 = avg_cost[index - 1, 0] / avg_cost[index - 2, 0]
            w_2 = avg_cost[index - 1, 3] / avg_cost[index - 2, 3]
            w_3 = avg_cost[index - 1, 6] / avg_cost[index - 2, 6]
            lambda_weight[0, index] = 3 * np.exp(w_1 / T) / (np.exp(w_1 / T) + np.exp(w_2 / T) + np.exp(w_3 / T))
            lambda_weight[1, index] = 3 * np.exp(w_2 / T) / (np.exp(w_1 / T) + np.exp(w_2 / T) + np.exp(w_3 / T))
            lambda_weight[2, index] = 3 * np.exp(w_3 / T) / (np.exp(w_1 / T) + np.exp(w_2 / T) + np.exp(w_3 / T))

    bar = Bar('Training', max=train_batch)

    # iteration for all batches
    model.train()
    nyuv2_train_dataset = iter(nyuv2_train_loader)
    for k in range(train_batch):
        train_data, train_label, train_depth, train_normal = nyuv2_train_dataset.next()
        train_data, train_label = train_data.cuda(), train_label.type(torch.LongTensor).cuda()
        train_depth, train_normal = train_depth.cuda(), train_normal.cuda()
        train_labels = {'semantic': train_label, 'depth': train_depth, 'normal': train_normal}
        train_pred, feat = model(train_data, feat=True)
        train_loss = get_dense_tasks_losses(train_pred, train_labels, opt.tasks, returndict=False)
        loss = 0
        norms = []
        w = torch.ones(len(tasks)).float().cuda()

        if opt.weight == 'mgda':
            W = feat
            W.retain_grad()
            gygw = []
            for i, t in enumerate(tasks):
                gygw.append(torch.autograd.grad(train_loss[i], W, retain_graph=True)[0].detach()/train_loss[i].data)
            sol, min_norm = MinNormSolver.find_min_norm_element(gygw)
            
            for i in range(len(tasks)):
                w[i] = float(sol[i])
                lambda_weight[i, index] = w[i].data
        elif opt.weight == 'gradnorm':
            norms = []
            # compute gradient w.r.t. last shared conv layer's parameters
            W = model.backbone.layer4[-1].conv3.weight

            for i, t in enumerate(tasks):
                gygw = torch.autograd.grad(train_loss[i], W, retain_graph=True)
                norms.append(torch.norm(torch.mul(Weights.weights[i], gygw[0])))
            norms = torch.stack(norms)
            task_loss = torch.stack(train_loss)
            if epoch ==0 and k == 0:
                initial_task_loss = task_loss
            loss_ratio = task_loss.data / initial_task_loss.data
            inverse_train_rate = loss_ratio / loss_ratio.mean()
            mean_norm = norms.mean()
            constant_term = mean_norm.data * (inverse_train_rate ** opt.alpha)
            grad_norm_loss = (norms - constant_term).abs().sum()
            w_grad = torch.autograd.grad(grad_norm_loss, Weights.weights)[0]
            for i in range(len(tasks)):
                w[i] = Weights.weights[i].data
                lambda_weight[i, index] = w[i].data
        elif opt.weight == 'dwa':
            for i in range(len(tasks)):
                w[i] = lambda_weight[i, index]
        elif opt.weight == 'uncert':
            logsigma = Weights.weights
            scalars = [1, 2, 2]
            loss = sum(1 / (scalars[i] * torch.exp(logsigma[i])) * train_loss[i] + logsigma[i] / 2 for i in range(len(tasks)))

            for i in range(len(tasks)):
                w[i] = 1 / (scalars[i] * torch.exp(logsigma[i])).item()
                lambda_weight[i, index] = w[i].data
        else:
            for i in range(len(tasks)):
                lambda_weight[i, index] = w[i].data
            loss = sum(w[i].data * train_loss[i] for i in range(len(tasks)))
        
        optimizer.zero_grad()
        if opt.weight == 'gs':
            optimizer.pc_backward(train_loss)
        elif opt.weight == 'imtl_l':
            optimizer.imtl_backward(train_loss, Weights.weights)
            for i in range(len(tasks)):
                w[i] = torch.exp(Weights.weights[i].data)
                lambda_weight[i, index] = w[i].data
        elif opt.weight == 'imtl_g':
            optimizer.imtl_backward(train_loss, Weights.weights, feat, 'imtl_g')
            for i in range(len(tasks)):
                w[i] = torch.tensor(1).to(train_loss[i].device)
                lambda_weight[i, index] = w[i].data
        elif opt.weight == 'imtl_h':
            optimizer.imtl_backward(train_loss, Weights.weights, feat, 'imtl_h')
            for i in range(len(tasks)):
                w[i] = torch.exp(Weights.weights[i].data)
                lambda_weight[i, index] = w[i].data
        elif opt.weight == 'gd':
            optimizer.gd_backward(train_loss)
            for i in range(len(tasks)):
                w[i] = torch.tensor(1).to(train_loss[i].device)
                lambda_weight[i, index] = w[i].data
        elif opt.weight == 'ca':
            optimizer.ca_backward(train_loss)
            for i in range(len(tasks)):
                w[i] = torch.tensor(1).to(train_loss[i].device)
                lambda_weight[i, index] = w[i].data
        else:
            loss.backward()
        if opt.weight == 'gradnorm':
            Weights.weights.grad = torch.zeros_like(Weights.weights.data)
            Weights.weights.grad.data = w_grad.data
        optimizer.step()
        
        if opt.weight == 'gradnorm':
            Weights.weights.data = len(tasks) * Weights.weights.data / Weights.weights.data.sum()

        cost[0] = train_loss[0].item()
        cost[1] = compute_miou(train_pred['semantic'], train_label, tasks_outputs['semantic']).item()
        cost[2] = compute_iou(train_pred['semantic'], train_label).item()
        cost[3] = train_loss[1].item()
        cost[4], cost[5] = depth_error(train_pred['depth'], train_depth)
        cost[6] = train_loss[2].item()
        cost[7], cost[8], cost[9], cost[10], cost[11] = normal_error(train_pred['normal'], train_normal)
        avg_cost[index, :12] += cost[:12] / train_batch
        bar.suffix  = '{} => ({batch}/{size}) | LossS: {loss_s:.4f} | LossD: {loss_d:.4f} | LossN: {loss_n:.4f} | Ws: {ws:.4f} | Wd: {wd:.4f}| Wn: {wn:.4f}'.format(
                    opt.weight,
                    batch=k + 1,
                    size=train_batch,
                    loss_s=cost[1],
                    loss_d=cost[3],
                    loss_n=cost[6],
                    ws=w[0].data,
                    wd=w[1].data,
                    wn=w[2].data,
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
            test_loss = get_dense_tasks_losses(test_pred, test_labels, opt.tasks)

            conf_mat.update(test_pred['semantic'].argmax(1).flatten(), test_label.flatten())
            depth_mat.update(test_pred['depth'], test_depth)
            normal_mat.update(test_pred['normal'], test_normal)
            cost[12] = test_loss[0].item()
            cost[15] = test_loss[1].item()
            cost[18] = test_loss[2].item()
            avg_cost[index, 12:] += cost[12:] / test_batch
        avg_cost[index, 13:15] = conf_mat.get_metrics()
        depth_metric = depth_mat.get_score()
        avg_cost[index, 16], avg_cost[index, 17] = depth_metric['l1'], depth_metric['rmse']
        normal_metric = normal_mat.get_score()
        avg_cost[index, 19], avg_cost[index, 20], avg_cost[index, 21], avg_cost[index, 22], avg_cost[index, 23] = normal_metric['mean'], normal_metric['rmse'], normal_metric['11.25'], normal_metric['22.5'], normal_metric['30']

    mtl_performance = 0.0
    mtl_performance += (avg_cost[index, 13]* 100 - stl_performance[opt.backbone+'_'+opt.head]['full']['semantic']) / stl_performance[opt.backbone+'_'+opt.head]['full']['semantic']
    mtl_performance -= (avg_cost[index, 16] - stl_performance[opt.backbone+'_'+opt.head]['full']['depth']) / stl_performance[opt.backbone+'_'+opt.head]['full']['depth']
    mtl_performance -= (avg_cost[index, 19] - stl_performance[opt.backbone+'_'+opt.head]['full']['normal']) / stl_performance[opt.backbone+'_'+opt.head]['full']['normal']
    mtl_performance = mtl_performance / len(tasks) * 100
    isbest = mtl_performance > best_performance
    print('current performance: {:.4f}, best performance: {:.4f}'.format(mtl_performance, best_performance))


    print('Epoch: {:04d} | TRAIN: {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} '
          'TEST: {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}'
          .format(index, avg_cost[index, 0], avg_cost[index, 1], avg_cost[index, 2], avg_cost[index, 3],
                avg_cost[index, 4], avg_cost[index, 5], avg_cost[index, 6], avg_cost[index, 7], avg_cost[index, 8], avg_cost[index, 9],
                avg_cost[index, 10], avg_cost[index, 11], avg_cost[index, 12], avg_cost[index, 13],
                avg_cost[index, 14], avg_cost[index, 15], avg_cost[index, 16], avg_cost[index, 17], avg_cost[index, 18],
                avg_cost[index, 19], avg_cost[index, 20], avg_cost[index, 21], avg_cost[index, 22], avg_cost[index, 23]))
    logger.append([index, avg_cost[index, 0], avg_cost[index, 1], avg_cost[index, 2], avg_cost[index, 3],
                avg_cost[index, 4], avg_cost[index, 5], avg_cost[index, 6], avg_cost[index, 7], avg_cost[index, 8], avg_cost[index, 9],
                avg_cost[index, 10], avg_cost[index, 11], avg_cost[index, 12], avg_cost[index, 13],
                avg_cost[index, 14], avg_cost[index, 15], avg_cost[index, 16], avg_cost[index, 17], avg_cost[index, 18],
                avg_cost[index, 19], avg_cost[index, 20], avg_cost[index, 21], avg_cost[index, 22], avg_cost[index, 23],
                lambda_weight[0, index], lambda_weight[1, index], lambda_weight[2, index]])

    if isbest:
        best_performance = mtl_performance
        print_index = index
    save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_performance': best_performance,
            'optimizer' : optimizer.state_dict(),
        }, isbest) 
print('Epoch: {:04d} | TRAIN: {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} '
          'TEST: {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}'
          .format(print_index, avg_cost[print_index, 0], avg_cost[print_index, 1], avg_cost[print_index, 2], avg_cost[print_index, 3],
                avg_cost[print_index, 4], avg_cost[print_index, 5], avg_cost[print_index, 6], avg_cost[print_index, 7], avg_cost[print_index, 8], avg_cost[print_index, 9],
                avg_cost[print_index, 10], avg_cost[print_index, 11], avg_cost[print_index, 12], avg_cost[print_index, 13],
                avg_cost[print_index, 14], avg_cost[print_index, 15], avg_cost[print_index, 16], avg_cost[print_index, 17], avg_cost[print_index, 18],
                avg_cost[print_index, 19], avg_cost[print_index, 20], avg_cost[print_index, 21], avg_cost[print_index, 22], avg_cost[print_index, 23]))
print('MTL Performance: {:2f}'.format(best_performance))