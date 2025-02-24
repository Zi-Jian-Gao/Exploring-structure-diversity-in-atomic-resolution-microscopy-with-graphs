import cv2
import torch
import random
from PIL import Image
import torch.nn as nn
from torchvision import transforms as T
import matplotlib.pyplot as plt
import numpy as np


# class RandomApply(nn.Module):
#     def __init__(self, fn, p):
#         super().__init__()
#         self.fn = fn
#         self.p = p
#     def forward(self, x):
#         if random.random() > self.p:
#             return x
#         return self.fn(x)
        

# class GaussianBlur(nn.Module):
#     def __init__(self, p):
#         super().__init__()
#         self.fn = T.GaussianBlur
#         self.p = p
        
#     def forward(self, x):
#         if random.random() > self.p:
#             return x
#         # random.choice([3, 5, 7, 9, 11])
#         aug = self.fn(kernel_size=random.choice([3, 5, 7, 9, 11]))
#         return aug(x)
        

# train_aug = torch.nn.Sequential(
#     # GaussianBlur(p=0.5),
#     T.Normalize(
#         mean=torch.tensor([0.485, 0.456, 0.406]),
#         std=torch.tensor([0.229, 0.224, 0.225])),
# )

# valid_aug = torch.nn.Sequential(
#     T.Normalize(
#         mean=torch.tensor([0.485, 0.456, 0.406]),
#         std=torch.tensor([0.229, 0.224, 0.225])),
# )

def train_aug(x):
    random = np.random.randint(-12, 12, 1)[0] / 255.
    return x + random