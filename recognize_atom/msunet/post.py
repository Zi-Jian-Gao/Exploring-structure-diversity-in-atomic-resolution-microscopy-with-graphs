import os
import cv2
import json
import glob
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import argparse
from PIL import Image
from skimage.measure import label
from skimage.measure import regionprops
from utils.labelme import save_pred_to_json,save_pred_to_json



if __name__ == '__main__':
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description='Process some images.')
    # 添加 --path 参数
    parser.add_argument('--save_path', type=str, required=True, help='Path to the images directory')
    parser.add_argument('--json_path', type=str, required=True, help='prediciton result')

    # 解析命令行参数
    args = parser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)
    save_pred_to_json(args.json_path, args.save_path)