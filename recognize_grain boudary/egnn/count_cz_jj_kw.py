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
from core.data import load_data_vor, find_cycles, get_graph
import torch
from torch_geometric.data import Data

def count( json_path_cz, jj_labels,point_lst_dict, psize=35):

    # img = cv2.imread(img_path, 0)
    points, edge_index, cz_labels, _ = load_data_cz(json_path_cz, max_edge_lengh=28)
    # cz原子序号拿到
    AA = np.where(cz_labels == 1)[0]

    cz_on_jj = np.sum(jj_labels[AA] == 2)
    # print(cz_on_jj)

    # cz原子坐标拿到
    points = points[AA]
    if np.size(AA) == 0:
        pass
    else:
        print(json_path)
        cz_num = len(AA)
        print(f"参杂的原子数量: {cz_num}")


    return cz_on_jj

import argparse
if __name__ == '__main__':
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description='Process some images.')
    # 添加 --path 参数
    parser.add_argument('--save_path', type=str, required=True, help='Path to the images directory')

    # 解析命令行参数
    args = parser.parse_args()

    json_lst = glob.glob('{}/*.json'.format(args.save_path), recursive=True); len(json_lst)
    total_type_1 = 0
    total_type_1_s = 0
    total_type_1_m = 0
    total_type_2 = 0
    total_type_2_s = 0
    total_type_2_m = 0
    total_cz = 0

    for json_path in (json_lst):
        print(json_path)
        base_name = json_path.split('/')[-1].split('.')[0]

        points, _, labels_jj, lights = load_data_jj(json_path,max_edge_lengh=28)

        cz_json_path = json_path.replace('gb', 'dophant_atom')

        points_cz, edge_index, cz_labels, _ = load_data_cz(cz_json_path, max_edge_lengh=28)

        # cz原子序号拿到
        AA = np.where(cz_labels == 1)[0]

        cz_num = len(AA)
        print(f"参杂的原子数量: {cz_num}")
        total_cz += cz_num

        cz_points = points[AA]

        #cz对应的jj类型
        cz_on_jj = labels_jj[AA]

        fei_jj = np.where(cz_on_jj == 1)[0]

        print(f"非JJ上参杂的原子数量: {len(fei_jj)}")

        fei_jj_cz_points = cz_points[fei_jj]

        jj = np.where(cz_on_jj == 2)[0]

        print(f"JJ上参杂的原子数量: {len(jj)}")

        jj_cz_points = cz_points[jj]


        kw_json_path = json_path.replace('gb', 'smov')

        points_kw, edge_index, labels_kw, _ = load_data_kw(kw_json_path, max_edge_lengh=28)

        # if np.size(fei_jj) == 0:
        #     print('无 非JJ参杂原子')
        # else:
        count_v = 0
        count_s = 0
        count_m = 0
        for point in fei_jj_cz_points:
            adj_point_label = []
            for idx, (s, e) in enumerate(edge_index.T):
                s = points[s]
                if s[0] == point[0] and s[1] == point[1]:
                    adj_point_label.append(labels_kw[e])
            adj_point_label = np.array(adj_point_label)
            if 3 in adj_point_label:
                count_v += 1
            else:
                if  np.sum(adj_point_label == 1) >=2:
                    count_s += 1
                if  np.sum(adj_point_label == 2) >=2:
                    count_m +=1
        print(f"非JJ上参杂旁有V的原子数量: {count_v}")
        print(f"非JJ上参杂旁有大于等于两个S的原子数量: {count_s}")
        print(f"非JJ上参杂旁有大于等于两个Mo的原子数量: {count_m}")


        total_type_1 += count_v
        total_type_1_s += count_s
        total_type_1_m += count_m



        count_v = 0
        count_s = 0
        count_m = 0
        for point in jj_cz_points:
            adj_point_label = []
            for idx, (s, e) in enumerate(edge_index.T):
                s = points[s]
                if s[0] == point[0] and s[1] == point[1]:
                    adj_point_label.append(labels_kw[e])
            adj_point_label = np.array(adj_point_label)
            if 3 in adj_point_label:
                count_v += 1
            else:
                if  np.sum(adj_point_label == 1) >=2:
                    count_s += 1
                if  np.sum(adj_point_label == 2) >=2:
                    count_m +=1
        print(f"JJ上参杂旁有V的原子数量: {count_v}")
        print(f"JJ上参杂旁有大于等于两个S的原子数量: {count_s}")
        print(f"JJ上参杂旁有大于等于两个Mo的原子数量: {count_m}")

        total_type_2 += count_v
        total_type_2_s += count_s
        total_type_2_m += count_m

    print(f"所有参杂原子数量: {total_cz}")
    print(f"所有非JJ上参杂旁有V的原子数量: {total_type_1}")
    print(f"所有JJ上参杂旁有V的原子数量: {total_type_2}")
    print(f"所有非JJ上参杂旁有大于等于两个S的原子数量: {total_type_1_s}")
    print(f"所有JJ上参杂旁有有大于等于两个S的原子数量: {total_type_2_s}")
    print(f"所有非JJ上参杂旁有大于等于两个Mo的原子数量: {total_type_1_m}")
    print(f"所有JJ上参杂旁有有大于等于两个Mo的原子数量: {total_type_2_m}")





