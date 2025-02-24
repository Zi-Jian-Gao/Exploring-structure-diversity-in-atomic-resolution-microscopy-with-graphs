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
from core.data import load_data_v2
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

    json_lst = glob.glob('{}/*.json'.format(args.save_path), recursive=True); len(json_lst)


    for json_path in (json_lst):
        base_name = json_path.split('/')[-1].split('.')[0]
        # points, edge_index, _, _ = load_data_v2(json_path,max_edge_lengh=27)
        points, edge_index, labels, _ = load_data_v2(json_path)
        # print(json_path.split('.')[0].split('dophant_atom')[0])
        # target_folder = os.path.join(json_path.split('.')[0].split('dophant_atom')[0], 'xj','raw')
        if 1 in labels:
            num = np.sum(labels)
            target_folder = '../../phase/raw'
            if not os.path.exists(target_folder):
                os.makedirs(target_folder)


            shutil.copy(json_path, target_folder)
            shutil.copy(json_path.replace('json','jpg'), target_folder)


