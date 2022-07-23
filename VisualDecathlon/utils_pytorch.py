#
# Authors: Wei-Hong Li
# This code is adapted from https://github.com/srebuffi/residual_adapters

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import os
import time
import numpy as np
from torch.autograd import Variable
import config_task
from tqdm import tqdm, trange
from time import sleep
import shutil
from utils import AverageMeter
import kornia

def adjust_learning_rate_and_learning_taks(optimizer, epoch, args, lr=None):
    """Sets the learning rate to the initial LR decayed by 10 every X epochs"""
    if lr is None:
        lr = args.lr
    if epoch >= args.step2: 
        lr = lr * 0.01
    elif epoch >= args.step1:
        lr = lr * 0.1
    else:
        lr = lr

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    print('lr: {}'.format(lr))

    # Return training classes
    return range(len(args.dataset))


# single domain training
def train(epoch, tloaders, tasks, net, args, optimizer, list_criterion=None):
    print('\nEpoch: %d' % epoch)
    net.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = [AverageMeter() for i in tasks]
    top1 = [AverageMeter() for i in tasks]
    end = time.time()
    
    loaders = [tloaders[i] for i in tasks]
    # min_len_loader = np.min([len(i) for i in loaders])
    min_len_loader = 10
    train_iter = [iter(i) for i in loaders]
        
    # for batch_idx in range(min_len_loader*len(tasks)):
    tbar = trange(min_len_loader*len(tasks), desc='Bar desc', leave=True)
    for batch_idx in tbar:
        config_task.first_batch = (batch_idx == 0)
        # Round robin process of the tasks
        current_task_index = batch_idx % len(tasks)
        inputs, targets = (train_iter[current_task_index]).next()
        config_task.task = tasks[current_task_index]
        # measure data loading time
        data_time.update(time.time() - end)
        if args.use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = args.criterion(outputs, targets)
        # measure accuracy and record loss
        (losses[current_task_index]).update(loss.data, targets.size(0))
        _, predicted = torch.max(outputs.data, 1)
        correct = predicted.eq(targets.data).cpu().float().sum()
        (top1[current_task_index]).update(correct*100./targets.size(0), targets.size(0))     
        # apply gradients   
        loss.backward()
        optimizer.step()
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        tbar.set_description('Epoch: [{0}] of task {1}, Time {batch_time.val:.3f} ({batch_time.avg:.3f}) Data {data_time.val:.3f} ({data_time.avg:.3f}) Loss {loss.avg:.3f}'.format(
                      epoch, args.dataset[0], batch_time=batch_time,
                      data_time=data_time, loss=(losses[current_task_index]))
            )
        tbar.refresh()
        sleep(0.01)
        
    for i in range(len(tasks)):
        print('Task {0} : Train Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Train Acc {top1.val:.3f} ({top1.avg:.3f})'.format(args.dataset[0], loss=losses[i], top1=top1[i]))

    return [top1[i].avg for i in range(len(tasks))], [losses[i].avg for i in range(len(tasks))]

# train vanilla multi-domain learning network
def trainmdl(epoch, tloaders, tasks, net, args, optimizer, list_criterion=None):
    print('\nEpoch: %d' % epoch)
    net.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = [AverageMeter() for i in tasks]
    top1 = [AverageMeter() for i in tasks]
    end = time.time()
    
    loaders = [tloaders[i] for i in tasks]
    num_batch = 100
    train_iter = [iter(i) for i in loaders]
        
    tbar = trange(num_batch, desc='Bar desc', leave=True)
    for batch_idx in tbar:
        inputs, targets = {}, {}
        num_samples = []
        for current_task_index, task in enumerate(args.dataset):
            try:
                inputs[task], targets[task] = (train_iter[current_task_index]).next()
            except:
                train_iter[current_task_index] = iter(loaders[current_task_index])
                inputs[task], targets[task] = (train_iter[current_task_index]).next()
            num_samples.append(targets[task].size(0))
        # measure data loading time
        data_time.update(time.time() - end)
        if args.use_cuda:
            for current_task_index, task in enumerate(args.dataset):
                inputs[task], targets[task] = inputs[task].cuda(), targets[task].cuda()
        optimizer.zero_grad()
        for task in args.dataset:
            if task not in ['imagenet12'] and 'imagenet12' in args.dataset:
                aug = kornia.augmentation.Resize((72, 72))
                inputs[task] = aug(inputs[task])
        outputs = net(torch.cat(list(inputs.values()), dim=0), num_samples=num_samples)

        loss = 0
        for current_task_index, task in enumerate(args.dataset):
            loss_ = args.criterion(outputs[current_task_index], targets[task])
            loss = loss + loss_
    
            (losses[current_task_index]).update(loss_.data, targets[task].size(0))
            _, predicted = torch.max(outputs[current_task_index].data, 1)
            correct = predicted.eq(targets[task].data).cpu().float().sum()
            (top1[current_task_index]).update(correct*100./targets[task].size(0), targets[task].size(0))   

        # apply gradients   
        loss.backward()
        optimizer.step()
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        description = 'Epoch: [{0}], Time {batch_time.avg:.3f} '.format(epoch, batch_time=batch_time)
        for current_task_index, task in enumerate(args.dataset):
            description += 'L{0}: {loss.avg:.3f} '.format(task[:2], loss=losses[current_task_index])

        tbar.set_description(description)
        tbar.refresh()
        sleep(0.01)

    for i in range(len(tasks)):
        print('Task {0} : Train Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Train Acc {top1.val:.3f} ({top1.avg:.3f})'.format(args.dataset[i], loss=losses[i], top1=top1[i]))

    return [top1[i].avg for i in range(len(tasks))], [losses[i].avg for i in range(len(tasks))]


# URL using cka and kl
def trainurl(epoch, tloaders, tasks, net, args, optimizer, adaptors, optimizer_adaptor, list_criterion=None):
    print('\nEpoch: %d' % epoch)
    net.train()
    adaptors.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = [AverageMeter() for i in tasks]
    kd_f_loss_tasks = [AverageMeter() for i in tasks]
    kd_p_loss_tasks = [AverageMeter() for i in tasks]
    top1 = [AverageMeter() for i in tasks]
    end = time.time()

    
    loaders = [tloaders[i] for i in tasks]
    num_batch = 100
    train_iter = [iter(i) for i in loaders]
        
    tbar = trange(num_batch, desc='Bar desc', leave=True)
    for batch_idx in tbar:
        inputs, targets = {}, {}
        num_samples = []
        for current_task_index, task in enumerate(args.dataset):
            try:
                inputs[task], targets[task] = (train_iter[current_task_index]).next()
            except:
                train_iter[current_task_index] = iter(loaders[current_task_index])
                inputs[task], targets[task] = (train_iter[current_task_index]).next()
            num_samples.append(targets[task].size(0))
        # measure data loading time
        data_time.update(time.time() - end)
        if args.use_cuda:
            for current_task_index, task in enumerate(args.dataset):
                inputs[task], targets[task] = inputs[task].cuda(), targets[task].cuda()
        optimizer.zero_grad()
        optimizer_adaptor.zero_grad()
        stl_feats, stl_logits = args.embed_many(inputs, return_type='list', kd=True, logits=True)
        for task in args.dataset:
            if task not in ['imagenet12'] and 'imagenet12' in args.dataset:
                aug = kornia.augmentation.Resize((72, 72))
                inputs[task] = aug(inputs[task])
        outputs, mtl_feats = net(torch.cat(list(inputs.values()), dim=0), kd=True, num_samples=num_samples)
        mtl_feats = adaptors(mtl_feats)
        loss = 0
        kd_f_loss = []
        kd_p_loss = []
        kd_f_loss_weights = {}
        for current_task_index, task in enumerate(args.dataset):
            loss_ = args.criterion(outputs[current_task_index], targets[task])
            loss = loss + loss_
            ft, fs = F.normalize(stl_feats[current_task_index], p=2, dim=1, eps=1e-12), F.normalize(mtl_feats[current_task_index], p=2, dim=1, eps=1e-12)
            kd_f_losses_ = args.criterion_dis(fs, ft.detach(), opt='kernelcka')
            kd_p_losses_ = args.criterion_div(outputs[current_task_index], stl_logits[current_task_index].detach())
            kd_weight = args.KD_F_LOSS_WEIGHTS[task] * args.kd_f_weight_annealing[task](t=epoch, opt='linear')
            bam_weight = args.KD_P_LOSS_WEIGHTS[task] * args.kd_p_weight_annealing[task](t=epoch, opt='linear')
    
            kd_f_loss_weights[task] = kd_weight
            if kd_weight > 0:
                kd_f_loss.append(kd_f_losses_ * kd_weight)
            if bam_weight > 0:
                kd_p_loss.append(kd_p_losses_ * bam_weight)

            (losses[current_task_index]).update(loss_.data, targets[task].size(0))
            (kd_f_loss_tasks[current_task_index]).update(kd_f_losses_.data, targets[task].size(0))
            (kd_p_loss_tasks[current_task_index]).update(kd_p_losses_.data, targets[task].size(0))
            _, predicted = torch.max(outputs[current_task_index].data, 1)
            correct = predicted.eq(targets[task].data).cpu().float().sum()
            (top1[current_task_index]).update(correct*100./targets[task].size(0), targets[task].size(0))   
        # Compute the loss with kd and bam loss
        if len(kd_f_loss) > 0:
            kd_f_loss = torch.stack(kd_f_loss).sum()
        else:
            kd_f_loss = 0
        if len(kd_p_loss) > 0:
            kd_p_loss = torch.stack(kd_p_loss).sum()
        else:
            kd_p_loss = 0
        
        loss = loss + args.sigma * kd_f_loss + args.beta * kd_p_loss

        # apply gradients   
        loss.backward()
        optimizer.step()
        optimizer_adaptor.step()
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        description = 'Epoch: [{0}], Time {batch_time.avg:.3f} '.format(epoch, batch_time=batch_time)
        for current_task_index, task in enumerate(args.dataset):
            description += 'L{0}: {loss.avg:.3f} '.format(task[:2], loss=losses[current_task_index])

        tbar.set_description(description)
        tbar.refresh()
        sleep(0.01)

    for i in range(len(tasks)):
        print('Task {0} : Train Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Train Acc {top1.val:.3f} ({top1.avg:.3f}) KD loss {kd_f_loss.avg:.3f} KL loss {kd_p_loss.avg:.3f} KD weight {kd_weight:.3f}'.format(args.dataset[i], loss=losses[i], top1=top1[i], kd_f_loss=kd_f_loss_tasks[i], kd_p_loss=kd_p_loss_tasks[i], kd_weight=list(kd_f_loss_weights.values())[i]))

    return [top1[i].avg for i in range(len(tasks))], [losses[i].avg for i in range(len(tasks))], [kd_f_loss_tasks[i].avg for i in range(len(tasks))], [kd_p_loss_tasks[i].avg for i in range(len(tasks))]



# URL using cka and kl and adapters
def trainurlad(epoch, tloaders, tasks, net, args, optimizer, adaptors, optimizer_adaptor, list_criterion=None):
    print('\nEpoch: %d' % epoch)
    net.train()

    adaptors.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = [AverageMeter() for i in tasks]
    kd_f_loss_tasks = [AverageMeter() for i in tasks]
    kd_p_loss_tasks = [AverageMeter() for i in tasks]
    top1 = [AverageMeter() for i in tasks]
    end = time.time()
    
    loaders = [tloaders[i] for i in tasks]
    num_batch = 100
    train_iter = [iter(i) for i in loaders]
        
    tbar = trange(num_batch, desc='Bar desc', leave=True)
    for batch_idx in tbar:
        inputs, targets = {}, {}
        num_samples = []
        for current_task_index, task in enumerate(args.dataset):
            try:
                inputs[task], targets[task] = (train_iter[current_task_index]).next()
            except:
                train_iter[current_task_index] = iter(loaders[current_task_index])
                inputs[task], targets[task] = (train_iter[current_task_index]).next()
            num_samples.append(targets[task].size(0))
        # measure data loading time
        data_time.update(time.time() - end)
        if args.use_cuda:
            for current_task_index, task in enumerate(args.dataset):
                inputs[task], targets[task] = inputs[task].cuda(), targets[task].cuda()
        optimizer.zero_grad()
        net.zero_grad()
        optimizer_adaptor.zero_grad()
        stl_feats, stl_logits = args.embed_many(inputs, return_type='list', kd=True, logits=True)

        loss = 0
        kd_f_loss = []
        kd_p_loss = []
        kd_f_loss_weights = {}
        for current_task_index, task in enumerate(args.dataset):
            config_task.task = tasks[current_task_index]
            mtl_feats = net.embed(inputs[task])
            outputs = net.cls_fn[config_task.task](mtl_feats)
            mtl_feats = adaptors.forward_task(mtl_feats, tasks[current_task_index])
            loss_ = args.criterion(outputs, targets[task])
            loss = loss + loss_

            ft, fs = F.normalize(stl_feats[current_task_index], p=2, dim=1, eps=1e-12), F.normalize(mtl_feats, p=2, dim=1, eps=1e-12)
            kd_f_losses_ = args.criterion_dis(fs, ft.detach(), opt='kernelcka')
            kd_p_losses_ = args.criterion_div(outputs, stl_logits[current_task_index].detach())

            kd_weight = args.KD_F_LOSS_WEIGHTS[task] * args.kd_f_weight_annealing[task](t=epoch, opt='linear')
            bam_weight = args.KD_P_LOSS_WEIGHTS[task] * args.bam_weight_annealing[task](t=epoch, opt='linear')

            kd_f_loss_weights[task] = kd_weight

            if kd_weight > 0:
                kd_f_loss.append(kd_f_losses_ * kd_weight)
            if bam_weight > 0:
                kd_p_loss.append(kd_p_losses_ * bam_weight)

            (losses[current_task_index]).update(loss_.data, targets[task].size(0))
            (kd_f_loss_tasks[current_task_index]).update(kd_f_losses_.data, targets[task].size(0))
            (kd_p_loss_tasks[current_task_index]).update(kd_p_losses_.data, targets[task].size(0))
            _, predicted = torch.max(outputs.data, 1)
            correct = predicted.eq(targets[task].data).cpu().float().sum()
            (top1[current_task_index]).update(correct*100./targets[task].size(0), targets[task].size(0))  
            task_loss = loss_
            if kd_weight>0:
                task_loss = task_loss + kd_f_losses_ * kd_weight * args.sigma
            if bam_weight>0:
                task_loss = task_loss + kd_p_losses_ * bam_weight * args.beta
            task_loss.backward()
        # Compute the loss with kd and bam loss
        if len(kd_f_loss) > 0:
            kd_f_loss = torch.stack(kd_f_loss).sum()
        else:
            kd_f_loss = 0
        if len(kd_p_loss) > 0:
            kd_p_loss = torch.stack(kd_p_loss).sum()
        else:
            kd_p_loss = 0
        loss = loss + args.sigma * kd_f_loss + args.beta * kd_p_loss
        # apply gradients   
        optimizer.step()
        optimizer_adaptor.step()
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        description = 'Epoch: [{0}], Time {batch_time.avg:.3f} '.format(epoch, batch_time=batch_time)
        for current_task_index, task in enumerate(args.dataset):
            description += 'L{0}: {loss.avg:.3f} '.format(task[:2], loss=losses[current_task_index])
        tbar.set_description(description)
        tbar.refresh()
        sleep(0.01)

    for i in range(len(tasks)):
        print('Task {0} : Train Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Train Acc {top1.val:.3f} ({top1.avg:.3f}) KD loss {kd_f_loss.avg:.3f} KL loss {kd_p_loss.avg:.3f} KD weight {kd_weight:.3f}'.format(args.dataset[i], loss=losses[i], top1=top1[i], kd_f_loss=kd_f_loss_tasks[i], kd_p_loss=kd_p_loss_tasks[i], kd_weight=list(kd_f_loss_weights.values())[i]))

    return [top1[i].avg for i in range(len(tasks))], [losses[i].avg for i in range(len(tasks))], [kd_f_loss_tasks[i].avg for i in range(len(tasks))], [kd_p_loss_tasks[i].avg for i in range(len(tasks))]

def test(epoch, loaders, all_tasks, net, best_acc, args, optimizer, ad=False):
    net.eval()
    losses = [AverageMeter() for i in all_tasks]
    top1 = [AverageMeter() for i in all_tasks]
    print('Epoch: [{0}]'.format(epoch))
    for itera in range(len(all_tasks)):
        i = all_tasks[itera]
        config_task.task = i
        for batch_idx, (inputs, targets) in enumerate(loaders[i]):
            if args.use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            with torch.no_grad():
                outputs = net(inputs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            loss = args.criterion(outputs, targets)
            
            losses[itera].update(loss.data, targets.size(0))
            _, predicted = torch.max(outputs.data, 1)
            correct = predicted.eq(targets.data).cpu().float().sum()
            top1[itera].update(correct*100./targets.size(0), targets.size(0))
        
        print('Task {0} : Test Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Test Acc {top1.val:.3f} ({top1.avg:.3f})'.format(args.dataset[i], loss=losses[itera], top1=top1[itera]))
    
    # Save checkpoint.
    acc = np.sum([top1[i].avg for i in range(len(all_tasks))])
    print('Saving..')
    state = {
        'net': net.state_dict(),
        'optimizer': optimizer.state_dict(),
        'acc': acc,
        'epoch': epoch,
    }
    if torch.__version__.lower().startswith('1.1'):
        torch.save(state, args.ckpdir+'/ckpt'+config_task.mode+''.join(args.dataset)+'_latest.pth')
    else:
        torch.save(state, args.ckpdir+'/ckpt'+config_task.mode+''.join(args.dataset)+'_latest.pth', _use_new_zipfile_serialization=False)
    
    if acc > best_acc:
        shutil.copyfile(args.ckpdir+'/ckpt'+config_task.mode+''.join(args.dataset)+'_latest.pth', args.ckpdir+'/ckpt'+config_task.mode+''.join(args.dataset)+'_best.pth')
        best_acc = acc
    # This is for training on train+val data. We use the best performing validation model
    # if it's obtained after 85th epoch or we use the model at 85th epoch
    if (epoch + 1) == 85 and args.trainval:
        shutil.copyfile(args.ckpdir+'/ckpt'+config_task.mode+''.join(args.dataset)+'_latest.pth', args.ckpdir+'/ckpt'+config_task.mode+''.join(args.dataset)+'_best.pth')
    
    return [top1[i].avg for i in range(len(all_tasks))], [losses[i].avg for i in range(len(all_tasks))], best_acc



def urltest(epoch, loaders, all_tasks, net, best_acc, args, optimizer, adaptors, optimizer_adaptor, ad=False):
    net.eval()
    losses = [AverageMeter() for i in all_tasks]
    top1 = [AverageMeter() for i in all_tasks]
    print('Epoch: [{0}]'.format(epoch))
    for itera in range(len(all_tasks)):
        i = all_tasks[itera]
        config_task.task = i
        for batch_idx, (inputs, targets) in enumerate(loaders[i]):
            if args.use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            with torch.no_grad():
                if ad:
                    feats = net.embed(inputs)
                    if isinstance(net.num_classes, list) and len(net.num_classes)>1:
                        outputs = net.cls_fn[all_tasks[itera]](feats)
                    else:
                        outputs = net.cls_fn(feats)
                else:
                    if inputs.size(2) == 64 and 'imagenet12' in args.dataset:
                        aug = kornia.augmentation.Resize((72, 72))
                        inputs = aug(inputs)
                    outputs = net(inputs, task=itera)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            loss = args.criterion(outputs, targets)
            
            losses[itera].update(loss.data, targets.size(0))
            _, predicted = torch.max(outputs.data, 1)
            correct = predicted.eq(targets.data).cpu().float().sum()
            top1[itera].update(correct*100./targets.size(0), targets.size(0))
        
        print('Task {0} : Test Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Test Acc {top1.val:.3f} ({top1.avg:.3f})'.format(args.dataset[i], loss=losses[itera], top1=top1[itera]))
    
    # Save checkpoint.
    acc = np.sum([top1[i].avg for i in range(len(all_tasks))])
    print('Saving..')
    state = {
        'net': net.state_dict(),
        'optimizer': optimizer.state_dict(),
        'adaptors': adaptors.state_dict(),
        'optimizer_adaptor': optimizer_adaptor.state_dict(),
        'acc': acc,
        'task_acc': [top1[i].avg for i in range(len(all_tasks))],
        'epoch': epoch,
    }
    if torch.__version__.lower().startswith('1.1'):
        torch.save(state, args.ckpdir+'/ckpt'+config_task.mode+''.join(args.dataset)+'_latest.pth')
    else:
        torch.save(state, args.ckpdir+'/ckpt'+config_task.mode+''.join(args.dataset)+'_latest.pth', _use_new_zipfile_serialization=False)
    if acc > best_acc:
        shutil.copyfile(args.ckpdir+'/ckpt'+config_task.mode+''.join(args.dataset)+'_latest.pth', args.ckpdir+'/ckpt'+config_task.mode+''.join(args.dataset)+'_best.pth')
        best_acc = acc

    return [top1[i].avg for i in range(len(all_tasks))], [losses[i].avg for i in range(len(all_tasks))], best_acc

def stlvalset(loaders, all_tasks, nets, args):
    # net.eval()
    losses = [AverageMeter() for i in all_tasks]
    top1 = [AverageMeter() for i in all_tasks]
    ans = {}
    for itera in range(len(all_tasks)):
        i = all_tasks[itera]
        config_task.task = i
        test_label_pred = []
        nets[i].eval()
        for batch_idx, (inputs, targets) in enumerate(loaders[i]):
            if args.use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            with torch.no_grad():
                outputs = nets[i](inputs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            loss = args.criterion(outputs, targets)
            
            losses[itera].update(loss.data, targets.size(0))
            _, predicted = torch.max(outputs.data, 1)
            correct = predicted.eq(targets.data).cpu().float().sum()
            top1[itera].update(correct*100./targets.size(0), targets.size(0))
        ans[args.dataset[i]] = test_label_pred
        print('Task {0} : Test Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Test Acc {top1.val:.3f} ({top1.avg:.3f})'.format(args.dataset[i], loss=losses[itera], top1=top1[itera]))
    return ans

def stltestset(loaders, all_tasks, nets, args):
    ans = {}
    for itera in range(len(all_tasks)):
        i = all_tasks[itera]
        config_task.task = i
        nets[i].eval()
        test_label_pred = []
        print('=> Predicting labels for dataset: {}'.format(args.dataset[i]))
        for batch_idx, (inputs, _) in enumerate(loaders[i]):
            if args.use_cuda:
                inputs = inputs.cuda()
            with torch.no_grad():
                outputs = nets[i](inputs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            _, predicted = torch.max(outputs.data, 1)
            predicted = predicted.cpu().numpy()
            test_label_pred.extend(predicted)
        ans[args.dataset[i]] = test_label_pred
        print('=> Evalation of dataset {} done!'.format(args.dataset[i]))
    
    return ans

def urltestset(loaders, all_tasks, net, args, ad=False):
    net.eval()
    ans = {}
    for itera in range(len(all_tasks)):
        i = all_tasks[itera]
        config_task.task = i
        test_label_pred = []
        print('=> Predicting labels for dataset: {}'.format(args.dataset[i]))
        for batch_idx, (inputs, _) in enumerate(loaders[i]):
            if args.use_cuda:
                inputs = inputs.cuda()
            with torch.no_grad():
                if ad:
                    feats = net.embed(inputs)
                    if isinstance(net.num_classes, list) and len(net.num_classes)>1:
                        outputs = net.cls_fn[all_tasks[itera]](feats)
                    else:
                        outputs = net.cls_fn(feats)
                else:
                    if inputs.size(2) == 64 and 'imagenet12' in args.dataset:
                        aug = kornia.augmentation.Resize((72, 72))
                        inputs = aug(inputs)
                    outputs = net(inputs, task=itera)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            
            _, predicted = torch.max(outputs.data, 1)
            predicted = predicted.cpu().numpy()
            test_label_pred.extend(predicted)
        ans[args.dataset[i]] = test_label_pred
        print('=> Evalation of dataset {} done!'.format(args.dataset[i]))
    return ans

def coco_results(ans, data_dir, imdb_dir, args, results_dir):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    import json
    class_name = args.dataset
    min_labs = {}
    for i in range(10):
        cocoGt = COCO('{}/{:s}_val.json'.format(imdb_dir, class_name[i]))
        imgIds = cocoGt.getImgIds()
        annIds = cocoGt.getAnnIds(imgIds=imgIds)
        anno = cocoGt.loadAnns(annIds)
        labels = [int(ann['category_id'])-1 for ann in anno]
        min_lab = min(labels)
        min_labs[class_name[i]] = min_lab
        


    res = []
    for i in range(10):
        cocoGt = COCO('{}/{:s}_test_stripped.json'.format(imdb_dir, class_name[i]))
        imgIds = sorted(cocoGt.getImgIds())
        cat = cocoGt.getCatIds()
        for item in imgIds:
            res.append({"image_id": item, "category_id": int(min_labs[class_name[i]] + 1 + ans[class_name[i]].pop(0))})
    with open('{}results.json'.format(results_dir), 'w') as outfile:
        json.dump(res, outfile)
    print('JSON FILE HAS BEEN CREATED. :D')


def coco_results_val(ans, data_dir, imdb_dir, args, results_dir):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    import json
    class_name = args.dataset
    min_labs = {}
    for i in range(10):
        cocoGt = COCO('{}/{:s}_val.json'.format(imdb_dir, class_name[i]))
        imgIds = cocoGt.getImgIds()
        annIds = cocoGt.getAnnIds(imgIds=imgIds)
        anno = cocoGt.loadAnns(annIds)
        labels = [int(ann['category_id'])-1 for ann in anno]
        min_lab = min(labels)
        min_labs[class_name[i]] = min_lab


    res = []
    for i in range(10):
        cocoGt = COCO('{}/{:s}_val.json'.format(imdb_dir, class_name[i]))
        imgIds = sorted(cocoGt.getImgIds())
        cat = cocoGt.getCatIds()
        for item in imgIds:
            res.append({"image_id": item, "category_id": int(min_labs[class_name[i]] + 1 + ans[class_name[i]].pop(0))})
    with open('{}results_val.json'.format(results_dir), 'w') as outfile:
        json.dump(res, outfile)
    print('JSON FILE HAS BEEN CREATED. :D')