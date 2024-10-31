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
from core.data import load_data_v2, load_data_cz
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
import shutil
import os

import argparse

if __name__ == '__main__':
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description='Process some images.')
    # 添加 --path 参数
    parser.add_argument('--save_path', type=str, required=True, help='Path to the images directory')

    # 解析命令行参数
    args = parser.parse_args()

    json_lst = glob.glob('{}/*.json'.format(args.save_path), recursive=True);
    len(json_lst)
    num_total_0 = 0
    num_total_1 = 0
    for json_path in (json_lst):
        print(json_path)
        base_name = json_path.split('/')[-1].split('.')[0]
        points, edge_index, labels, _ = load_data_v2(json_path, max_edge_lengh=28)

        cz_json_path = json_path.replace('phase', 'dophant_atom')

        points_cz, edge_index, cz_labels, _ = load_data_cz(cz_json_path, max_edge_lengh=28)

        # cz原子序号拿到
        AA = np.where(cz_labels == 1)[0]
        # 对应原子的xj标签拿到
        select_labels = labels[AA]
        num_0 = np.sum((select_labels == 0))
        num_total_0 += num_0
        num_1 = np.sum((select_labels == 1))
        num_total_1 += num_1
        print('掺杂在1相界的个数:'+ str(num_1))
        print('掺杂在0/2相界的个数:'+ str(num_0))

    print('合计掺杂在1相界的个数:'+ str(num_total_1))
    print('合计掺杂在0/2相界的个数:'+ str(num_total_0))



