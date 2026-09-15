import os
import sys


import torch
import torch.nn as nn
import torch.nn.functional as F

# Application modules
sys.path.insert(0, os.path.abspath('../'))


def py_sigmoid_focal_loss(pred,
                          target,
                          weight=None,
                          gamma=2.0,
                          alpha=0.25,
                          reduction='mean',
                          avg_factor=None):
    pred_sigmoid = pred.sigmoid()
    target = target.type_as(pred)
    pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
    focal_weight = (alpha * target + (1 - alpha) *
                    (1 - target)) * pt.pow(gamma)
    loss = F.binary_cross_entropy_with_logits(
        pred, target, reduction='none') * focal_weight
    if weight is not None:
        if weight.ndim == loss.ndim:
            loss = loss * weight
        else:
            loss = loss.view(-1, 1) * weight.view(-1, 1)
    if reduction == 'mean':
        if avg_factor is not None:
            loss = loss.sum() / avg_factor
        else:
            loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()
    return loss


def sigmoid_focal_loss(pred,
                       target,
                       weight=None,
                       gamma=2.0,
                       alpha=0.25,
                       reduction='mean',
                       avg_factor=None):
    return py_sigmoid_focal_loss(pred, target, weight, gamma, alpha, reduction, avg_factor)

class FocalLoss(nn.Module):

    def __init__(self,
                 use_sigmoid=True,
                 gamma=2.0,
                 alpha=0.25,
                 reduction='mean',
                 loss_weight=1.0):
        super(FocalLoss, self).__init__()
        assert use_sigmoid is True, 'Only sigmoid focal loss supported now.'
        self.use_sigmoid = use_sigmoid
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        '''
        print(pred.shape)
      
        print(target.shape)
        print(target)
        print(target.nonzero())
        idx = target.nonzero()
        print(target[idx])
        '''
#        print(weight,self.gamma,self.alpha,reduction,avg_factor)
        if self.use_sigmoid:
            loss_cls = self.loss_weight * sigmoid_focal_loss(
                pred,
                target,
                weight,
                gamma=self.gamma,
                alpha=self.alpha,
                reduction=reduction,
                avg_factor=avg_factor)
        else:
            raise NotImplementedError
        return loss_cls
                                         
