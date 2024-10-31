import os
import cv2
import glob
import json
import shutil
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from PIL import Image
from labelme import utils
from utils.densemap import get_dotsmap
from skimage.measure import label
from skimage.measure import regionprops

class_dict = {
    1: 'Norm', 
    2: 'SV',
    3: 'LineSV',
}

class_dict_rev = {
    'Norm': 1, 
    'SV': 2,
    'LineSV': 3,
}

def get_json(img_path, lbl):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    w, h = np.where(lbl != 0)
    
    points = [[h[i], w[i]] for i in range(len(h))]
    points = np.array(points, np.int16).tolist()
    shapes = [{"label": class_dict[lbl[item[1], item[0]]], "points": [item], "group_id": None, "shape_type": "point", "flags": {}} for item in points]
    imagePath = img_path.split('/')[-1]
    imageData = utils.img_arr_to_b64(img).decode('utf-8')
    imageHeight, imageWidth = img.shape
    
    json_data = {
        'version': '5.0.1',
        'flags': {},
        'shapes': shapes,
        'imagePath': imagePath,
        'imageData': imageData,
        'imageHeight': imageHeight,
        'imageWidth': imageWidth
    }
    
    return json_data


def get_json2(img_path, lbl):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    w, h = np.where(lbl != 0)

    points = [[h[i], w[i]] for i in range(len(h))]
    points = np.array(points, np.int16).tolist()
    shapes = [{"label": class_dict[lbl[item[1], item[0]]], "points": [item[::-1]], "group_id": None, "shape_type": "point",
               "flags": {}} for item in points]
    imagePath = img_path.split('/')[-1]
    imageData = utils.img_arr_to_b64(img).decode('utf-8')
    imageHeight, imageWidth = img.shape

    json_data = {
        'version': '5.0.1',
        'flags': {},
        'shapes': shapes,
        'imagePath': imagePath,
        'imageData': imageData,
        'imageHeight': imageHeight,
        'imageWidth': imageWidth
    }

    return json_data

def hough_center_detection(i, rp, labeled_img, img_size=2048):
    hs, ws, he, we = rp.bbox
    hs = np.clip(hs - 8, 0, img_size-1)
    ws = np.clip(ws - 8, 0, img_size-1)
    he = np.clip(he + 8, 0, img_size-1)
    we = np.clip(we + 8, 0, img_size-1)

    m = np.array(labeled_img == rp.label, np.uint8)[hs:he, ws:we]
    m = cv2.dilate(m, np.ones((3, 3)))  
    
    cricles = cv2.HoughCircles(
        m,
        method = cv2.HOUGH_GRADIENT,
        dp = 1,
        minDist = 13,
        # minDist = 18,
        minRadius = 5,
        # minRadius = 7,
        maxRadius = 12,
        param1 = 5,
        # param1 = 8,
        param2 = 6,
        # param2 = 8,
    )
    
    if cricles is None:
        return np.array([])
    
    if (rp.area > 400) & (cricles.shape[1] != 2):
        print(i, cricles.shape[1])
    
    centers = np.round(cricles[0][:, :2][:, ::-1] + [hs, ws])
    centers = np.array(centers, np.int32)
    
    return centers

#
# def get_mask(probs, min_size=16, ds=5):
#     binary = np.array(probs * 255., np.uint8)
#     _, binary = cv2.threshold(binary, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#
#     centers = []
#     mask = np.zeros(binary.shape)
#
#     labeled_img = label(binary)
#     rps = regionprops(labeled_img, intensity_image=probs)
#
#     for rp in rps:
#         if rp.area < min_size:
#             continue
#
#         h, w = np.array(np.round(rp.centroid), np.int32)
#         mask[h, w] = 1
#
#     return mask


def get_mask_v2(probs, min_area=48):
    binary = np.array(probs * 255., np.uint8)
    _, binary = cv2.threshold(binary, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    centers = []
    mask = np.zeros(binary.shape)
    
    labeled_img = label(binary)
    rps = regionprops(labeled_img, intensity_image=probs)

    for i, rp in enumerate(rps):
        if rp.area < min_area:
            continue

        rp_centers = hough_center_detection(i, rp, labeled_img)

        if len(rp_centers) == 0:
            h, w = np.array(np.round(rp.centroid), np.int32)
            mask[h, w] = 1
        else:
            for h, w in rp_centers:
                if h < mask.shape[0] and w < mask.shape[1]:  # 确保索引在 mask 的范围内
                    if h == mask.shape[0]:  # mask 的最后一行
                        continue
                    if w == mask.shape[1]:  # mask 的最后一列
                        continue
                    mask[h, w] = 1
    return mask



def save_pred_to_json(json_path, save_path):
    with open(json_path) as f:
        data = json.load(f)
    
    pred = np.array(data['pred'])
    lb = np.array(data['label'], dtype=np.uint8)
    img_path = np.array(data['img_path'])
    nums = len(img_path)

    for i in range(nums):
        name = img_path[i].split('/')[-1].split('.')[0]

        # gt
        # Image.fromarray(lb[i]).save('{}/{}.png'.format(save_path, name))
        # Image.open(img_path[i]).save('{}/{}.jpg'.format(save_path, name))
        
        # pred
        pred_lbl = get_mask_v2(pred[i])
        json_data = get_json(img_path[i], pred_lbl)
        
        with open('{}/{}.json'.format(save_path, name), 'w') as f:
            json.dump(json_data, f)


def save_pred_to_json_jj(json_path, save_path):
    with open(json_path) as f:
        data = json.load(f)

    pred = np.array(data['pred'])
    lb = np.array(data['label'], dtype=np.uint8)
    img_path = np.array(data['img_path'])
    nums = len(img_path)

    for i in range(nums):
        name = img_path[i].split('/')[-1].split('.')[0]

        # gt
        Image.fromarray(lb[i]).save('{}/{}.png'.format(save_path, name))
        Image.open(img_path[i]).save('{}/{}.jpg'.format(save_path, name))

        # pred
        pred_lbl = get_mask_v2(pred[i])
        json_data = get_json(img_path[i], pred_lbl)

        with open('{}/{}.json'.format(save_path, name), 'w') as f:
            json.dump(json_data, f)
