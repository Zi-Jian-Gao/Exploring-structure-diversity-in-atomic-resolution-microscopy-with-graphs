import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, x, y, eps=1):
        N = y.size(0)

        x_flat = x.view(N, -1)
        y_flat = y.view(N, -1)

        inter = (x_flat * y_flat).sum(1)
        union = x_flat.sum(1) + y_flat.sum(1)
        
        dice_coefficient = 2. * (inter + eps) / (union + eps)
        loss = 1. - dice_coefficient.sum() / N

        return loss


class MyLoss(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.bceloss = nn.BCELoss()
        self.diceloss = DiceLoss()
        self.weights = weights
        
    def forward(self, x, y):
        loss = 0.
        for i in range(len(x)):
            loss += (self.bceloss(x[i], y[i]) + self.diceloss(x[i], y[i])) * self.weights[i]
        
        return loss
    

def mce(y_pred, y_true):
    x = torch.sum(y_pred[3], dim=[1, 2, 3])
    y = torch.sum(y_true[3], dim=[1, 2, 3])
    
    return torch.mean(torch.abs(x - y) / 100.)


def iou(y_pred, y_true, eps=1.):
    y_pred = y_pred[-1] > 0.5
    y_true = y_true[-1]
    iou_score = torch.mean((y_pred * y_true).sum((1, 2, 3)) / (torch.Tensor((y_true + y_pred) != 0.).sum((1, 2, 3)) + eps))
    
    return iou_score


def dice(y_pred, y_true, eps=1.):
    y_pred = y_pred[-1] > 0.5
    y_true = y_true[-1]
    dice_score = 2 * torch.mean((y_pred * y_true).sum((1, 2, 3)) / (y_true.sum((1, 2, 3)) + y_pred.sum((1, 2, 3)) + eps))
    
    return dice_score
