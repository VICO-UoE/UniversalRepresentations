#
# Authors: Wei-Hong Li

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Variable
import torch.nn.init as init
import numpy as np
import pdb

def get_dense_tasks_losses(outputs, labels, tasks, returndict=False, opt=None):
    losses = {}
    for task in tasks:
        losses[task] = get_task_loss(outputs[task], labels[task], task, opt)
    if returndict:
        return losses
    else:
        return list(losses.values())

def get_task_loss(output, label, task, opt=None):
    if task == 'semantic':
        # semantic loss: depth-wise cross entropy
        loss = F.nll_loss(output, label, ignore_index=-1)
        return loss

    if task == 'depth':
        # binary mark to mask out undefined pixel space
        binary_mask = (torch.sum(label, dim=1) != 0).type(torch.FloatTensor).unsqueeze(1).cuda()

        # depth loss: l1 norm
        loss = torch.sum(torch.abs(output - label) * binary_mask) / torch.nonzero(binary_mask).size(0)
        return loss
    if task == 'normal':
        if opt is None:
            binary_mask = (torch.sum(label, dim=1) != 0).type(torch.FloatTensor).unsqueeze(1).cuda()
            # normal loss: dot product
            loss = 1 - torch.sum((output * label) * binary_mask) / torch.nonzero(binary_mask).size(0)
        elif opt == 'l1':
            valid_mask = (torch.sum(label, dim=1, keepdim=True) != 0).cuda()
            loss = torch.sum(F.l1_loss(output, label, reduction='none').masked_select(valid_mask)) \
                / torch.nonzero(valid_mask, as_tuple=False).size(0)
        return loss

def get_performances(output, label, task, tasks_outputs):
    if task == 'semantic':
        miou = compute_miou(output, label, tasks_outputs['semantic'])
        iou = compute_iou(output, label)
        return miou, iou
    elif task == 'depth':
        return depth_error(output, label)
    elif task == 'normal':
        return normal_error(output, label)

def compute_miou(x_pred, x_output, class_nb):
    _, x_pred_label = torch.max(x_pred, dim=1)
    x_output_label = x_output
    batch_size = x_pred.size(0)
    for i in range(batch_size):
        true_class = 0
        first_switch = True
        for j in range(class_nb):
            pred_mask = torch.eq(x_pred_label[i], j * torch.ones(x_pred_label[i].shape).type(torch.LongTensor).cuda())
            true_mask = torch.eq(x_output_label[i], j * torch.ones(x_output_label[i].shape).type(torch.LongTensor).cuda())
            mask_comb = pred_mask.type(torch.FloatTensor) + true_mask.type(torch.FloatTensor)
            union     = torch.sum((mask_comb > 0).type(torch.FloatTensor))
            intsec    = torch.sum((mask_comb > 1).type(torch.FloatTensor))
            if union == 0:
                continue
            if first_switch:
                class_prob = intsec / union
                first_switch = False
            else:
                class_prob = intsec / union + class_prob
            true_class += 1
        if i == 0:
            batch_avg = class_prob / true_class
        else:
            batch_avg = class_prob / true_class + batch_avg
    return batch_avg / batch_size

def compute_iou(x_pred, x_output):
    _, x_pred_label = torch.max(x_pred, dim=1)
    x_output_label = x_output
    batch_size = x_pred.size(0)
    for i in range(batch_size):
        if i == 0:
            pixel_acc = torch.div(torch.sum(torch.eq(x_pred_label[i], x_output_label[i]).type(torch.FloatTensor)),
                        torch.sum((x_output_label[i] >= 0).type(torch.FloatTensor)))
        else:
            pixel_acc = pixel_acc + torch.div(torch.sum(torch.eq(x_pred_label[i], x_output_label[i]).type(torch.FloatTensor)),
                        torch.sum((x_output_label[i] >= 0).type(torch.FloatTensor)))
    return pixel_acc / batch_size

def depth_error(x_pred, x_output):
    binary_mask = (torch.sum(x_output, dim=1) != 0).unsqueeze(1).cuda()
    x_pred_true = x_pred.masked_select(binary_mask)
    x_output_true = x_output.masked_select(binary_mask)
    abs_err = torch.abs(x_pred_true - x_output_true)
    rel_err = torch.abs(x_pred_true - x_output_true) / x_output_true
    return torch.sum(abs_err) / torch.nonzero(binary_mask).size(0), torch.sum(rel_err) / torch.nonzero(binary_mask).size(0)

def normal_error(x_pred, x_output):
    binary_mask = (torch.sum(x_output, dim=1) != 0)
    error = torch.acos(torch.clamp(torch.sum(x_pred * x_output, 1).masked_select(binary_mask), -1, 1)).detach().cpu().numpy()
    error = np.degrees(error)
    return np.mean(error), np.median(error), np.mean(error < 11.25), np.mean(error < 22.5), np.mean(error < 30)
