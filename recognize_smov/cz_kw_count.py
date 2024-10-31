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
from core.data import load_data_jj,load_data_cz,load_data_kw
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
import shutil
import os


def count(json_path_cz, kw_labels):

    points, edge_index, cz_labels, _ = load_data_cz(json_path_cz, max_edge_lengh=28)

    AA = np.where(cz_labels == 1)[0]
    # cz原子坐标拿到
    cz_points = points[AA]
    if np.size(AA) == 0:
        pass
    else:
        print(json_path)
        cz_num = len(AA)
        print(f"参杂的原子数量: {cz_num}")
        count_v = 0
        count_s = 0
        count_m = 0
        for point in cz_points:

            adj_point_label = []
            for idx, (s, e) in enumerate(edge_index.T):
                s = points[s]
                if s[0] == point[0] and s[1] == point[1]:
                    adj_point_label.append(kw_labels[e])
            adj_point_label = np.array(adj_point_label)
            if 3 in adj_point_label:
                count_v +=1
            else:
                if  np.sum(adj_point_label == 1) >=2:
                    count_s += 1
                if  np.sum(adj_point_label == 2) >=2:
                    count_m +=1
        print(f"参杂旁有V的原子数量: {count_v}")
        print(f"参杂旁有大于等于两个S的原子数量: {count_s}")
        print(f"参杂旁有大于等于两个Mo的原子数量: {count_m}")


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

    for json_path in (json_lst):
        base_name = json_path.split('/')[-1].split('.')[0]
        points, edge_index, labels, _ = load_data_kw(json_path,max_edge_lengh=28)

        cz_json_path = json_path.replace('smov', 'dophant_atom')

        count(cz_json_path, labels)
