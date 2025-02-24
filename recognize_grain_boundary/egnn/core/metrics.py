import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score

# def f1(y_pred, y_true):
#     y_pred = torch.argmax(y_pred, axis=1)
#     score = f1_score(y_true.cpu().numpy(), y_pred.cpu().numpy())
    
#     return score


def acc(y_pred, y_true):
    y_pred = torch.argmax(y_pred, axis=1)
    score = accuracy_score(y_true.cpu().numpy(), y_pred.cpu().numpy())
    
    return score

class LabelSmoothingCrossEntropy(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.05):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, x, target):
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()