import torch
import torch.nn as nn
from Nail_dataset import get_dataset
from torch.utils.data import DataLoader
from Model import get_model
import matplotlib.pyplot as plt
import cv2
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from torch.nn.functional import binary_cross_entropy
import torch
from torch.utils.tensorboard import SummaryWriter


class FocalLoss(nn.Module):
    """
    Focal Loss for Dense Object Detection [https://arxiv.org/abs/1708.02002]
    Digest the paper as below:
        α, balances the importance of positive/negative examples
        γ, focusing parameter that controls the strength of the modulating term
            CE(pt) = −log(pt) ==> pt = exp(-CE)
            FL(pt) = −α((1 − pt)^γ) * log(pt)
        In general α should be decreased slightly as γ is increased (for γ = 2, α = 0.25 works best).
    """
    def __init__(self, focusing_param=2, balance_param=0.25):
        super().__init__()
        self.gamma = focusing_param
        self.alpha = balance_param

    def forward(self, inputs, targets, weights=None):
        logpt = -binary_cross_entropy(inputs, targets, weights)
        pt = torch.exp(logpt)
        # compute the loss
        focal_loss = -((1-pt)**self.gamma) * logpt
        balanced_focal_loss = self.alpha * focal_loss
        return balanced_focal_loss

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=0.00001):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
#         inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1-dice
    
class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super().__init__()

    def forward(self, inputs, targets):
        smooth = 0
        num = targets.size(0) # number of batches
#         print(num)
        m1 = inputs.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = (m1 * m2)
#         print(intersection)
        score = (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) - intersection.sum(1) + smooth)
        iou = score.sum() / num
        # three kinds of loss formulas: (1) 1 - iou (2) -iou (3) -torch.log(iou)
        return 1. - iou
    
    
    
class IoUScore(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super().__init__()

    def forward(self, inputs, targets):
        smooth = 0.00001
        num = targets.size(0) # number of batches
#         print(num)
        m1 = inputs.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = (m1 * m2)
#         print(intersection)
        score = (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) - intersection.sum(1) + smooth)
        iou = score.sum() / num
        # three kinds of loss formulas: (1) 1 - iou (2) -iou (3) -torch.log(iou)
        return iou