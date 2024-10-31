import os
import cv2
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import albumentations as A
import scipy.ndimage as ndimage
import torchvision.transforms as T

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from core.slidespliter import SlideSpliter,SlideSpliter_unequal

# define heavy augmentations
def get_training_augmentation(dim):
    train_transform = [
        A.CenterCrop(dim, dim),
        A.HorizontalFlip(),
        A.VerticalFlip(),
        A.GaussianBlur(),
        A.RandomRotate90(),
        A.RandomBrightnessContrast(),
        A.ShiftScaleRotate(),
        # A.PadIfNeeded(min_height=dim, min_width=dim, always_apply=True, border_mode=0),
        A.PadIfNeeded(min_height=dim, min_width=dim, always_apply=True),
        A.Normalize(),
    ]

    return A.Compose(train_transform)

def get_validation_augmentation(dim):
    test_transform = [
        A.CenterCrop(dim, dim),
        A.PadIfNeeded(min_height=dim, min_width=dim, always_apply=True, border_mode=0, value=0),
        A.Normalize(),
    ]

    return A.Compose(test_transform)


def get_validation_augmentation_v2(h,w):
    test_transform = [
        A.CenterCrop(h, w),
        A.PadIfNeeded(min_height=h, min_width=w, always_apply=True, border_mode=0, value=0),
        A.Normalize(),
    ]

    return A.Compose(test_transform)


class MyDataset(Dataset):
    def __init__(self, img_list, dim=256, sigma=3, data_type='train'):
        self.images = img_list
        self.dots = [item.replace('/img/', '/lbl/') for item in self.images]
        self.aug = get_training_augmentation(dim) if data_type == 'train' else get_validation_augmentation(dim)
        self.sigma = sigma
        self.data_type = data_type
        self.sample_weights = np.sqrt([int(item.split('_')[-1].split('.')[0])+1 for item in img_list])
        self.sample_weights = [int(item.split('_')[-1].split('.')[0])+1 for item in img_list]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        image = cv2.imread(self.images[i], cv2.IMREAD_COLOR)

        if self.data_type != 'test':
            mask = cv2.imread(self.dots[i], cv2.IMREAD_GRAYSCALE)
        else:
            mask = np.zeros(image.shape[:2])

        if self.aug:
            sample = self.aug(image=image, mask=mask)
            image, mask = sample['image'].transpose(2, 0, 1), sample['mask']

        h, w = np.where(mask != 0); lbls = mask[h, w]; points = np.array([w, h]).transpose()
        ps = mask.shape[0]
        d4 = np.zeros((ps, ps))
        for idx, point in enumerate(points):
            cv2.circle(d4, point, 8, 1, cv2.FILLED)

        d3 = np.zeros((int(ps/2), int(ps/2)))
        for idx, point in enumerate(points):
            cv2.circle(d3, np.array(np.round(point/2.), np.int16), 4, 1, cv2.FILLED)

        d2 = np.zeros((int(ps/4), int(ps/4)))
        for idx, point in enumerate(points):
            cv2.circle(d2, np.array(np.round(point/4.), np.int16), 2, 1, cv2.FILLED)

        d1 = np.zeros((int(ps/8), int(ps/8)))
        for idx, point in enumerate(points):
            cv2.circle(d1, np.array(np.round(point/8.), np.int16), 1, 1, cv2.FILLED)

        d1 = d1[np.newaxis, :, :]
        d2 = d2[np.newaxis, :, :]
        d3 = d3[np.newaxis, :, :]
        d4 = d4[np.newaxis, :, :]

        return np.array(image, np.float32), [np.array(d1, np.float32), np.array(d2, np.float32), np.array(d3, np.float32), np.array(d4, np.float32)]


class MyDatasetSlide(Dataset):
    def __init__(self, img_list, ps=256, dim=2048):
        self.images = img_list
        self.dots = [item.replace('/img/', '/lbl/') for item in self.images]
        self.spliter = SlideSpliter(patch_size=ps)
        self.aug = get_validation_augmentation(dim)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        image = cv2.imread(self.images[i], cv2.IMREAD_COLOR)

        if os.path.exists(self.dots[i]):
            mask = cv2.imread(self.dots[i], cv2.IMREAD_GRAYSCALE)
        else:
            print('Skip Load Slide Label!')
            mask = np.zeros(image.shape[:2])

        image = self.aug(image=image)['image']
        image = self.spliter.split(image).transpose(0, 3, 1, 2)

        return np.array(image, np.float32), np.array(mask, np.float32)


class MyDatasetSlide_test(Dataset):
    def __init__(self, img_list, ps=256, dim=2048):
        self.images = img_list
        self.dots = [item.replace('/img/', '/lbl/') for item in self.images]
        self.spliter = SlideSpliter(slide_size=dim,patch_size=ps,roi_size=64)
        self.aug = get_validation_augmentation(dim)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        image = cv2.imread(self.images[i], cv2.IMREAD_COLOR)

        if os.path.exists(self.dots[i]):
            mask = cv2.imread(self.dots[i], cv2.IMREAD_GRAYSCALE)
        else:
            print('Skip Load Slide Label!')
            mask = np.zeros(image.shape[:2])

        image = self.aug(image=image)['image']
        image = self.spliter.split2(image).transpose(0, 3, 1, 2)

        return np.array(image, np.float32), np.array(mask, np.float32)


class MyDatasetSlide_test_uneuqal(Dataset):
    def __init__(self, img_list, ps=256, roi=64):
        self.images = img_list
        self.dots = [item.replace('/img/', '/lbl/') for item in self.images]
        self.spliter = SlideSpliter_unequal(patch_size=ps,roi_size=roi)


    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        image = cv2.imread(self.images[i], cv2.IMREAD_COLOR)

        if os.path.exists(self.dots[i]):
            mask = cv2.imread(self.dots[i], cv2.IMREAD_GRAYSCALE)
        else:
            print('Skip Load Slide Label!')
            mask = np.zeros(image.shape[:2])

        h, w, _ = image.shape
        self.aug = get_validation_augmentation_v2(h,w)
        image = self.aug(image=image)['image']
        # image = self.spliter.split2(image).transpose(0, 3, 1, 2)
        image = self.spliter.split2(image)

        return np.array(image, np.float32), np.array(mask, np.float32)
