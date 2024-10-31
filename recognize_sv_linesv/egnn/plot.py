import os
import cv2
import glob
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from PIL import Image
from core.data import load_data
from core.data import load_data_v2
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
import argparse


def plot2(img_folder, json_folder, psize=26, output_folder='output'):
    c = ['#9BB6CF', '#76F1A2', '#EDC08C', 'red']

    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历指定文件夹中的所有文件
    for img_filename in os.listdir(img_folder):
        if img_filename.endswith(".jpg"):  # 确保是jpg文件
            img_path = os.path.join(img_folder, img_filename)
            json_filename = img_filename.replace('.jpg', '.json')
            json_path = os.path.join(json_folder, json_filename)

            img = cv2.imread(img_path, 0)

            points, edge_index, labels, _ = load_data_v2(json_path)

            mask_pd = np.zeros((2048, 2048))
            # mask_pd = np.zeros((512, 512))
            mask_pd[points[:, 0], points[:, 1]] = labels + 1
            mask_pd = np.array(mask_pd, np.uint8)

            plt.figure(figsize=(9, 9))
            plt.imshow(img, cmap='gray')
            for i in range(4):
                h, w = np.where(mask_pd == i + 1)
                plt.scatter(w, h, s=psize, c=c[i])


            plt.axis('off')

            plt.tight_layout()

            # 保存图像
            output_path = os.path.join(img_filename)
            plt.savefig(output_path, dpi = 300)
            plt.close()  # 关闭当前的绘图窗口，以避免内存泄漏



if __name__ == '__main__':
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description='Process some images.')
    # 添加 --path 参数
    parser.add_argument('--result_path', type=str, required=True, help='Path to the images directory')

    # 解析命令行参数
    args = parser.parse_args()

    plot2(args.result_path, args.result_path)