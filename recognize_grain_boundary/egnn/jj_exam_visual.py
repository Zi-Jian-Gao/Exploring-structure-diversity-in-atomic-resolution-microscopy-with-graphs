import os
import cv2
import glob
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from PIL import Image
from utils.e2e_metrics import get_metrics
from core.data import load_data_jj,load_data_cz
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
import shutil
import os


def plot(img_path, json_path_cz, jj_labels, target_folder, psize=35):
    img = cv2.imread(img_path, 0)
    points, edge_index, cz_labels, _ = load_data_cz(json_path_cz, max_edge_lengh=28)

    plt.figure(figsize=(9, 9))
    plt.imshow(img, cmap='gray')

    mask_pd_xj = np.zeros(img.shape)
    mask_pd_xj[points[:, 0], points[:, 1]] = jj_labels  #1,2
    mask_pd_xj = np.array(mask_pd_xj, np.uint8)

    c = ['orange', 'blue']
    for i in range(2):
        h, w = np.where(mask_pd_xj == i + 1)
        plt.scatter(w, h, s=psize, c=c[i])

    for idx, (s, e) in enumerate(edge_index.T):
        s = points[s]
        e = points[e]
        plt.plot([s[1], e[1]], [s[0], e[0]], linewidth=1, c='#C0C0C0', zorder=1)


    mask_pd_cz = np.zeros(img.shape)
    mask_pd_cz[points[:, 0], points[:, 1]] = cz_labels
    mask_pd_cz = np.array(mask_pd_cz, np.uint8)
    h, w = np.where(mask_pd_cz == 1)
    plt.scatter(w, h, s=psize, c='red')

    base_name = json_path_cz.split('/')[-1].split('.')[0]
    file = os.path.join(target_folder, base_name + '_visual.png')


    plt.axis('off')
    plt.tight_layout()
    plt.savefig(file, dpi=300)
    plt.close()


import argparse
if __name__ == '__main__':
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description='Process some images.')
    # 添加 --path 参数
    parser.add_argument('--save_path', type=str, required=True, help='Path to the images directory')

    # 解析命令行参数
    args = parser.parse_args()

    json_lst = glob.glob('{}/*.json'.format(args.save_path), recursive=True); len(json_lst)

    for json_path in (json_lst):
        base_name = json_path.split('/')[-1].split('.')[0]
        points, edge_index, labels, _ = load_data_jj(json_path,max_edge_lengh=28)
        img_path = json_path.replace('json', 'jpg')
        cz_json_path = json_path.replace('gb', 'dophant_atom')

        target_folder = '../../smov/raw'
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)
        shutil.copy(json_path, target_folder)
        shutil.copy(img_path, target_folder)


        target_folder = '../../gb_visual/'
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)

        plot(img_path, cz_json_path, labels, target_folder)

        # shutil.copy(json_path, target_folder)
        # shutil.copy(img_path, target_folder)


