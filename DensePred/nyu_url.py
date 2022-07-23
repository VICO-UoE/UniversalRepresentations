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
from models.get_model import get_model, get_stl_model
from utils.evaluation import ConfMatrix, DepthMeter, NormalsMeter
from utils.dense_losses import get_dense_tasks_losses, get_task_loss, compute_miou, compute_iou, depth_error, normal_error
import numpy as np
from progress.bar import Bar as Bar
from utils import Logger, AverageMeter, accuracy, mkdir_p, savefig

parser = argparse.ArgumentParser(description='Knowledge Distillation for Multi-task (SegNet)')
parser.add_argument('--weight', default='uniform', type=str, help='multi-task weighting: uniform, gradnorm, mgda, uncert, dwa, gs, etc')
parser.add_argument('--backbone', default='segnet', type=str, help='shared backbone')
parser.add_argument('--head', default='segnet_head', type=str, help='task-specific decoder')
parser.add_argument('--tasks', default='semantic', nargs='+', help='Task(s) to be trained')
parser.add_argument('--pretrained', dest='pretrained', action='store_true', help='using pretrained weight from ImageNet')
parser.add_argument('--dilated', dest='dilated', action='store_true', help='Dilated')
parser.add_argument('--fuse_hrnet', dest='fuse_hrnet', action='store_true', help='fuse_hrnet')
parser.add_argument('--method', default='vanilla', type=str, help='vanilla or mtan')
parser.add_argument('--dataroot', default='nyuv2', type=str, help='dataset root')
parser.add_argument('--alr', default=1e-2, type=float, help='initial learning rate')
parser.add_argument('--out', default='./results/url', help='Directory to output the result')
parser.add_argument('--single-dir', default='./results/stl', help='Directory to output the result')
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


class adaptor(torch.nn.Module):
    def __init__(self):
        super(adaptor, self).__init__()
        self.conv1 = torch.nn.Conv2d(512, 64, 1, bias=False)
        self.conv2 = torch.nn.Conv2d(64, 64, 1, bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)

            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

    def forward(self, inputs):
        results = []
        results.append(self.conv1(inputs[0]))
        results.append(self.conv2(inputs[1]))
        return results

class Weight(torch.nn.Module):
    def __init__(self, tasks):
        super(Weight, self).__init__()
        self.weights = nn.Parameter(torch.FloatTensor([1.0, 1.0, 1.0]))


def save_checkpoint(state, is_best, checkpoint=opt.out, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, '{}_{}_url_{}_{}_alr_{}_'.format(opt.backbone, opt.head, opt.method, opt.weight, opt.alr) + filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, '{}_{}_url_{}_{}_alr_{}_'.format(opt.backbone, opt.head, opt.method, opt.weight, opt.alr) + 'model_best.pth.tar'))

if not os.path.isdir(opt.out):
    mkdir_p(opt.out)
title = 'NYUv2'
logger = Logger(os.path.join(opt.out, '{}_{}_url_{}_{}_alr_{}_'.format(opt.backbone, opt.head, opt.method, opt.weight, opt.alr) + 'log.txt'), title=title)
logger.set_names(['Epoch', 'T.Ls', 'T. mIoU', 'T. Pix', 'T.Ld', 'T.abs', 'T.rel', 'T.Ln', 'T.Mean', 'T.Med', 'T.11', 'T.22', 'T.30',
    'V.Ls', 'V. mIoU', 'V. Pix', 'V.Ld', 'V.abs', 'V.rel', 'V.Ln', 'V.Mean', 'V.Med', 'V.11', 'V.22', 'V.30', 'ds', 'dd', 'dh'])

# define model, optimiser and scheduler
model = get_model(opt, tasks_outputs=tasks_outputs).cuda()
Weights = Weight(tasks).cuda()
single_model = {}
adaptors = {}

for i, t in enumerate(tasks):
    single_model[i] = get_stl_model(opt, tasks_outputs, t).cuda()
    checkpoint = torch.load('{}/{}_{}_stl_{}_model_best.pth.tar'.format(opt.single_dir, opt.backbone, opt.head, tasks[i]))
    single_model[i].load_state_dict(checkpoint['state_dict'], strict=False)
    adaptors[i] = adaptor().cuda()

params = []
params += model.parameters()
params += [Weights.weights]
optimizer = optim.Adam(params, lr=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
if 'uncert' in opt.weight:
    Weights.weights.data *= -0.5
params = []
total_epoch = 200
for i in range(len(tasks)):
    params += adaptors[i].parameters()
adaptor_optimizer = optim.Adam(params, lr=opt.alr, weight_decay=5e-4)
adaptor_scheduler = optim.lr_scheduler.CosineAnnealingLR(adaptor_optimizer, total_epoch)

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
    shuffle=True, num_workers=0)


# define parameters
total_epoch = 200
train_batch = len(nyuv2_train_loader)
test_batch = len(nyuv2_test_loader)
avg_cost = np.zeros([total_epoch, 24], dtype=np.float32)
# best_performance = 100
best_performance = -100
isbest = False
for epoch in range(total_epoch):
    index = epoch
    cost = np.zeros(24, dtype=np.float32)
    dist_loss_save = {}
    for i, t in enumerate(tasks):
        dist_loss_save[i] = AverageMeter()

    # apply Dynamic Weight Average

    bar = Bar('Training', max=train_batch)

    # iteration for all batches
    model.train()
    loss_weights ={}
    for task in tasks:
        loss_weights[task] = AverageMeter()
    nyuv2_train_dataset = iter(nyuv2_train_loader)
    for k in range(train_batch):
        # pdb.set_trace()
        train_data, train_label, train_depth, train_normal = nyuv2_train_dataset.next()
        train_data, train_label = train_data.cuda(), train_label.type(torch.LongTensor).cuda()
        train_depth, train_normal = train_depth.cuda(), train_normal.cuda()
        train_labels = {'semantic': train_label, 'depth': train_depth, 'normal': train_normal}

        train_pred, feat_s = model(train_data, feat=True)
        train_loss = get_dense_tasks_losses(train_pred, train_labels, opt.tasks)
        loss = 0

        w = torch.ones(len(tasks)).cuda()
        
        dist_loss = []
        for i, t in enumerate(tasks):
            with torch.no_grad():
                feat_ti = single_model[i].embed(train_data)
            feat_si = feat_s

            if isinstance(feat_ti, list):
                feat_ti = feat_ti[-1]
            feat_ti = F.normalize(feat_ti, p=2, dim=1, eps=1e-12)
            feat_si = adaptors[i](feat_si)
            dist_feat = torch.tensor(0).cuda()
            for l in range(len(feat_si)):
                if feat_si[l].size()[2:] != feat_ti.size()[2:]:
                    feat_si[l] = F.interpolate(feat_si[l], feat_ti.size()[2:] , mode='bilinear')
                feat_si[l] = F.normalize(feat_si[l], p=2, dim=1, eps=1e-12)
                dist_feat = dist_feat + (feat_si[l] - feat_ti.detach()).pow(2).sum(1).mean()
            dist_loss.append(dist_feat)
            dist_loss_save[i].update(dist_loss[i].data.item(), train_data.size(0))

        # loss weights (\lambda) for distillation losses
        lambda_ = [1, 1, 2]
        dist_loss = sum(dist_loss[i] * lambda_[i] for i in range(len(tasks)))

        if 'uniform' in opt.weight:
            loss = torch.mean(sum(w[i] * train_loss[i] for i in range(len(tasks))))
        elif 'uncert' in opt.weight:
            logsigma = Weights.weights
            scalars = [1, 2, 2]
            loss = sum(1 / (scalars[i] * torch.exp(logsigma[i])) * train_loss[i] + logsigma[i] / 2 for i in range(len(tasks)))

        for i, task in enumerate(tasks):
            if 'uncert' in opt.weight:
                w[i] = 1 / (scalars[i] * torch.exp(logsigma[i])).item()
            loss_weights[task].update(w[i].data, 1)

        loss = loss + dist_loss

        optimizer.zero_grad()
        adaptor_optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        adaptor_optimizer.step()

        

        cost[0] = train_loss[0].item()
        cost[1] = compute_miou(train_pred['semantic'], train_label, tasks_outputs['semantic']).item()
        cost[2] = compute_iou(train_pred['semantic'], train_label).item()
        cost[3] = train_loss[1].item()
        cost[4], cost[5] = depth_error(train_pred['depth'], train_depth)
        cost[6] = train_loss[2].item()
        cost[7], cost[8], cost[9], cost[10], cost[11] = normal_error(train_pred['normal'], train_normal)
        avg_cost[index, :12] += cost[:12] / train_batch
        
        bar.suffix  = '{} => ({batch}/{size}) | LossS: {loss_s:.3f} | LossD: {loss_d:.3f} | LossN: {loss_n:.3f} | Ds: {ds:.2f} | Dd: {dd:.2f}| Dn: {dn:.2f} | Ws: {ws:.2f} | Wd: {wd:.2f}| Wn: {wn:.2f}'.format(
                'url',
                batch=k + 1,
                size=train_batch,
                loss_s=cost[1],
                loss_d=cost[3],
                loss_n=cost[6],
                ds=dist_loss_save[0].val,
                dd=dist_loss_save[1].val,
                dn=dist_loss_save[2].val,
                ws=loss_weights['semantic'].avg,
                wd=loss_weights['depth'].avg,
                wn=loss_weights['normal'].avg,
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
    scheduler.step()
    adaptor_scheduler.step()
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
                dist_loss_save[0].avg, dist_loss_save[1].avg, dist_loss_save[2].avg])

    if isbest:
        best_performance = mtl_performance
        print_index = index
    save_dict = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_performance': best_performance,
            'optimizer' : optimizer.state_dict(),
            'adaptor_optimizer' : adaptor_optimizer.state_dict(),
        }
    for i in range(len(tasks)):
        save_dict['adaptor_{}'.format(tasks[i])] = adaptors[i].state_dict()

    save_checkpoint(save_dict, isbest) 
print('Epoch: {:04d} | TRAIN: {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} '
              'TEST: {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} '
              .format(print_index, avg_cost[print_index, 0], avg_cost[print_index, 1], avg_cost[print_index, 2], avg_cost[print_index, 3],
                avg_cost[print_index, 4], avg_cost[print_index, 5], avg_cost[print_index, 6], avg_cost[print_index, 7], avg_cost[print_index, 8], avg_cost[print_index, 9],
                avg_cost[print_index, 10], avg_cost[print_index, 11], avg_cost[print_index, 12], avg_cost[print_index, 13],
                avg_cost[print_index, 14], avg_cost[print_index, 15], avg_cost[print_index, 16], avg_cost[print_index, 17], avg_cost[print_index, 18],
                avg_cost[print_index, 19], avg_cost[print_index, 20], avg_cost[print_index, 21], avg_cost[print_index, 22], avg_cost[print_index, 23]))
print('MTL Performance: {:2f}'.format(best_performance))