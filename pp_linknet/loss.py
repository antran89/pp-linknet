import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import cv2
import numpy as np
class dice_bce_loss(nn.Module):
    def __init__(self, batch=True):
        super(dice_bce_loss, self).__init__()
        self.batch = batch
        self.bce_loss = nn.BCELoss()
        
    def soft_dice_coeff(self, y_pred, y_true):
        smooth = 0.0  # may change
        if self.batch:
            i = torch.sum(y_true)
            j = torch.sum(y_pred)
            intersection = torch.sum(y_true * y_pred)
        else:
            i = y_true.sum(1).sum(1).sum(1)
            j = y_pred.sum(1).sum(1).sum(1)
            intersection = (y_true * y_pred).sum(1).sum(1).sum(1)
        score = (2. * intersection + smooth) / (i + j + smooth)
        #score = (intersection + smooth) / (i + j - intersection + smooth)#iou
        return score.mean()

    def soft_dice_loss(self, y_pred, y_true):
        loss = 1 - self.soft_dice_coeff(y_pred, y_true)
        return loss
        
    def __call__(self, y_pred, y_true):
        a =  self.bce_loss(y_pred, y_true)
        b =  self.soft_dice_loss(y_pred, y_true)
        return a + b
    
class dice_bce_loss_fp16(nn.Module):
    def __init__(self, batch=True):
        super(dice_bce_loss_fp16, self).__init__()
        self.batch = batch
        self.bce_loss = nn.BCEWithLogitsLoss()

    def soft_dice_coeff(self, y_pred, y_true):
        smooth = 0.0  # may change
        if self.batch:
            i = torch.sum(y_true)
            j = torch.sum(y_pred)
            intersection = torch.sum(y_true * y_pred)
        else:
            i = y_true.sum(1).sum(1).sum(1)
            j = y_pred.sum(1).sum(1).sum(1)
            intersection = (y_true * y_pred).sum(1).sum(1).sum(1)
        score = (2. * intersection + smooth) / (i + j + smooth)
        #score = (intersection + smooth) / (i + j - intersection + smooth)#iou
        return score.mean()

    def soft_dice_loss(self, y_pred, y_true):
        y_pred = torch.sigmoid(y_pred)
        loss = 1 - self.soft_dice_coeff(y_pred, y_true)
        return loss

    def __call__(self, y_pred, y_true):
        a =  self.bce_loss(y_pred, y_true)
        b =  self.soft_dice_loss(y_pred, y_true)
        return a + b

class FocalLoss(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.num_classes = num_classes

    def get_weight(self,x,t):
        alpha,gamma = 0.25,0.5
        p = x.sigmoid()
        pt = p*t + (1-p)*(1-t)
        w = alpha*t + (1-alpha)*(1-t)
        return w * (1-pt).pow(gamma)

    def forward(self, preds, targets):
        w = self.get_weight(preds, targets)# for the last part
        weight = Variable(w, requires_grad=False)
        return F.binary_cross_entropy_with_logits(preds, targets, weight=weight, reduction='mean') / self.num_classes

class dice_focal_loss(nn.Module):
    def __init__(self, num_classes=1, batch=False):
        super(dice_focal_loss, self).__init__()
        self.batch = batch
        self.focal_loss = FocalLoss(num_classes=num_classes)

    def soft_dice_loss(self, y_pred, y_true):
        smooth = 1e-5  # may change
        if self.batch:
            i = torch.sum(y_true*y_true)
            j = torch.sum(y_pred*y_pred)
            intersection = torch.sum(y_true * y_pred)
        else:
            i = (y_true*y_true).sum(1).sum(1).sum(1)
            j = (y_pred*y_pred).sum(1).sum(1).sum(1)
            intersection = (y_true * y_pred).sum(1).sum(1).sum(1)
        score = (2. * intersection + smooth) / (i + j + smooth)
        #score = (intersection + smooth) / (i + j - intersection + smooth)#iou
        score = 1 - score
        return score.mean()
    
    # y_pred is logits of network output
    def __call__(self, y_pred, y_true):
        a =  self.focal_loss(y_pred, y_true)
        b =  self.soft_dice_loss(y_pred.sigmoid(), y_true)
        #print(5*a.item(), b.item())
        return 5*a + b

class dice_loss(nn.Module):
    def __init__(self, batch=False):
        super(dice_loss, self).__init__()
        self.batch = batch
        
    def soft_dice_loss(self, y_pred, y_true):
        smooth = 1e-5  # may change
        if self.batch:
            i = torch.sum(y_true*y_true)
            j = torch.sum(y_pred*y_pred)
            intersection = torch.sum(y_true * y_pred)
        else:
            i = (y_true*y_true).sum(1).sum(1).sum(1)
            j = (y_pred*y_pred).sum(1).sum(1).sum(1)
            intersection = (y_true * y_pred).sum(1).sum(1).sum(1)
        score = (2. * intersection + smooth) / (i + j + smooth)
        #score = (intersection + smooth) / (i + j - intersection + smooth)#iou
        score = 1 - score
        return score.mean()

    def __call__(self, y_pred, y_true):
        return self.soft_dice_loss(y_pred.sigmoid(), y_true)
    
class dice_bce_loss2(nn.Module):
    """
    Dice BCE with correct loss function.
    """
    def __init__(self):
        super(dice_bce_loss2, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.soft_dice_loss = dice_loss()

    def __call__(self, y_pred, y_true):
        a = self.bce_loss(y_pred, y_true)
        b = self.soft_dice_loss(y_pred.sigmoid(), y_true)
        return a + b

class dice_loss2(nn.Module):
    def __init__(self, batch=False):
        super(dice_loss2, self).__init__()
        self.batch = batch

    def soft_dice_loss(self, y_pred, y_true):
        smooth = 1e-5  # may change
        if self.batch:
            i = torch.sum(y_true)
            j = torch.sum(y_pred)
            intersection = torch.sum(y_true * y_pred)
        else:
            i = y_true.sum(1).sum(1).sum(1)
            j = y_pred.sum(1).sum(1).sum(1)
            intersection = (y_true * y_pred).sum(1).sum(1).sum(1)
        score = (2. * intersection + smooth) / (i + j + smooth)
        #score = (intersection + smooth) / (i + j - intersection + smooth)#iou
        score = 1 - score
        return score.mean()

    def __call__(self, y_pred, y_true):
        return self.soft_dice_loss(y_pred.sigmoid(), y_true)

class bce_with_logits_loss(nn.Module):
    def __init__(self):
        super(bce_with_logits_loss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def __call__(self, y_pred, y_true):
        return self.bce_loss(y_pred, y_true)
