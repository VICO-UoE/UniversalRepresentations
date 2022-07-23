#
# Authors: Wei-Hong Li

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pdb
import numpy as np
import copy
import random


class IMTL():
    def __init__(self, optimizer, reduction='mean'):
        self._optim, self._reduction = optimizer, reduction
        return

    @property
    def optimizer(self):
        return self._optim

    def zero_grad(self):
        '''
        clear the gradient of the parameters
        '''

        return self._optim.zero_grad(set_to_none=True)

    def step(self):
        '''
        update the parameters with the gradient
        '''

        return self._optim.step()

    def imtl_backward(self, objectives, weights, shared_feat=None, opt=None, retain_graph=False):
        '''
        calculate the gradient of the parameters

        input:
        - objectives: a list of objectives
        - weights: a list of loss weights s
        '''
        if shared_feat is None:
            self._imtl_l(objectives, weights, retain_graph=retain_graph)
        else:
            if opt is None or opt == 'imtl_h':
                obj = [objectives[i] * torch.exp(weights[i]) - weights[i] for i in range(len(objectives))]
                self._imtl_g(obj, shared_feat)
            elif opt == 'imtl_g':
                self._imtl_g(objectives, shared_feat)
            else: exit('invalid balanced method')
        return

    def _imtl_l(self, objectives, weights, retain_graph=False):
        obj = sum(objectives[i] * torch.exp(weights[i]) - weights[i] for i in range(len(objectives)))
        obj.backward(retain_graph=retain_graph)
        return
    def _imtl_g(self, objectives, shared_feat):
        # objectives = [objectives[i] * torch.exp(weights[i]) - weights[i] for i in range(len(objectives))]
        # if isinstance(shared_feat, list):
        glgf = []
        ut = []
        for i in range(len(objectives)):
            if isinstance(shared_feat, list):
                glgf.append(torch.autograd.grad(objectives[i], shared_feat[i], retain_graph=True)[0].detach().mean(0).view(1,-1))
            else:
                glgf.append(torch.autograd.grad(objectives[i], shared_feat, retain_graph=True)[0].detach().mean(0).view(1,-1))
            ut.append(F.normalize(glgf[i], p=2, dim=-1, eps=1e-12))
        glgf = torch.cat(glgf, dim=0)
        ut = torch.cat(ut, dim=0)
        D = glgf[0][None,:] - glgf[1:]
        U = ut[0][None,:] - ut[1:]
        alpha = torch.mm(glgf[0][None,:], U.t())
        alpha = torch.mm(alpha, torch.inverse(torch.mm(D, U.t())))
        alpha_ = [1 - alpha.sum().data]
        for i in range(1, len(objectives)):
            alpha_.append(alpha[0, i-1].data)
        grads, shapes, has_grads = self._pack_grad(objectives)
        balanced_grads = self._aggregate_grads(grads, has_grads, alpha_)
        balanced_grads = self._unflatten_grad(balanced_grads, shapes[0])
        self._set_grad(balanced_grads)
        return
    
    def _aggregate_grads(self, grads, has_grads, alpha, shapes=None):
        shared = torch.stack(has_grads).prod(0).bool()
        merged_grad = torch.zeros_like(grads[0]).to(grads[0].device)
        if self._reduction:
            merged_grad[shared] = torch.stack([g[shared] * a
                                           for g, a in zip(grads, alpha)]).mean(dim=0)
        elif self._reduction == 'sum':
            merged_grad[shared] = torch.stack([g[shared] * a
                                           for g, a in zip(grads, alpha)]).sum(dim=0)
        else: exit('invalid reduction method')

        merged_grad[~shared] = torch.stack([g[~shared]
                                            for g in grads]).sum(dim=0)
        return merged_grad

    def _project_conflicting(self, grads, has_grads, shapes=None):
        shared = torch.stack(has_grads).prod(0).bool()
        pc_grad, num_task = copy.deepcopy(grads), len(grads)
        for g_i in pc_grad:
            random.shuffle(grads)
            for g_j in grads:
                g_i_g_j = torch.dot(g_i, g_j)
                if g_i_g_j < 0:
                    g_i -= (g_i_g_j) * g_j / (g_j.norm()**2)
        merged_grad = torch.zeros_like(grads[0]).to(grads[0].device)
        if self._reduction:
            merged_grad[shared] = torch.stack([g[shared]
                                           for g in pc_grad]).mean(dim=0)
        elif self._reduction == 'sum':
            merged_grad[shared] = torch.stack([g[shared]
                                           for g in pc_grad]).sum(dim=0)
        else: exit('invalid reduction method')

        merged_grad[~shared] = torch.stack([g[~shared]
                                            for g in pc_grad]).sum(dim=0)
        return merged_grad

    def _set_grad(self, grads):
        '''
        set the modified gradients to the network
        '''

        idx = 0
        for group in self._optim.param_groups:
            for p in group['params']:
                # if p.grad is None: continue
                p.grad = grads[idx]
                idx += 1
        return

    def _pack_grad(self, objectives):
        '''
        pack the gradient of the parameters of the network for each objective
        
        output:
        - grad: a list of the gradient of the parameters
        - shape: a list of the shape of the parameters
        - has_grad: a list of mask represent whether the parameter has gradient
        '''

        grads, shapes, has_grads = [], [], []
        for obj in objectives:
            self._optim.zero_grad(set_to_none=True)
            obj.backward(retain_graph=True)
            grad, shape, has_grad = self._retrieve_grad()
            grads.append(self._flatten_grad(grad, shape))
            has_grads.append(self._flatten_grad(has_grad, shape))
            shapes.append(shape)
        return grads, shapes, has_grads

    def _unflatten_grad(self, grads, shapes):
        unflatten_grad, idx = [], 0
        for shape in shapes:
            length = np.prod(shape)
            unflatten_grad.append(grads[idx:idx + length].view(shape).clone())
            idx += length
        return unflatten_grad

    def _flatten_grad(self, grads, shapes):
        flatten_grad = torch.cat([g.flatten() for g in grads])
        return flatten_grad

    def _retrieve_grad(self):
        '''
        get the gradient of the parameters of the network with specific 
        objective
        
        output:
        - grad: a list of the gradient of the parameters
        - shape: a list of the shape of the parameters
        - has_grad: a list of mask represent whether the parameter has gradient
        '''

        grad, shape, has_grad = [], [], []
        for group in self._optim.param_groups:
            for p in group['params']:
                # if p.grad is None: continue
                # tackle the multi-head scenario
                if p.grad is None:
                    shape.append(p.shape)
                    grad.append(torch.zeros_like(p).to(p.device))
                    has_grad.append(torch.zeros_like(p).to(p.device))
                    continue
                shape.append(p.grad.shape)
                grad.append(p.grad.clone())
                has_grad.append(torch.ones_like(p).to(p.device))
        return grad, shape, has_grad
    def state_dict(self):
        return self._optim.state_dict()


class TestNet(nn.Module):
    def __init__(self):
        super().__init__()
        self._linear = nn.Linear(3, 4)

    def forward(self, x):
        return self._linear(x)


class MultiHeadTestNet(nn.Module):
    def __init__(self):
        super().__init__()
        self._linear = nn.Linear(3, 2)
        self._head1 = nn.Linear(2, 4)
        self._head2 = nn.Linear(2, 4)

    def forward(self, x):
        feat = self._linear(x)
        return self._head1(feat), self._head2(feat)


if __name__ == '__main__':

    # fully shared network test
    torch.manual_seed(4)
    x, y = torch.randn(2, 3), torch.randn(2, 4)
    net = TestNet()
    y_pred = net(x)
    pc_adam = PCGrad(optim.Adam(net.parameters()))
    pc_adam.zero_grad()
    loss1_fn, loss2_fn = nn.L1Loss(), nn.MSELoss()
    loss1, loss2 = loss1_fn(y_pred, y), loss2_fn(y_pred, y)

    pc_adam.pc_backward([loss1, loss2])
    for p in net.parameters():
        print(p.grad)

    print('-' * 80)
    # seperated shared network test

    torch.manual_seed(4)
    x, y = torch.randn(2, 3), torch.randn(2, 4)
    net = MultiHeadTestNet()
    y_pred_1, y_pred_2 = net(x)
    pc_adam = PCGrad(optim.Adam(net.parameters()))
    pc_adam.zero_grad()
    loss1_fn, loss2_fn = nn.MSELoss(), nn.MSELoss()
    loss1, loss2 = loss1_fn(y_pred_1, y), loss2_fn(y_pred_2, y)

    pc_adam.pc_backward([loss1, loss2])
    for p in net.parameters():
        print(p.grad)

