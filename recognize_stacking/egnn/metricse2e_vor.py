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
from core.data import get_y_3
from core.data import load_data_v2
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
import argparse

class_dict = {
    0: '0',
    1: '1',
    2: '2',
}

if __name__ == '__main__':
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description='Process some images.')
    # 添加 --path 参数
    parser.add_argument('--save_path', type=str, required=True, help='Path to the images directory')
    parser.add_argument('--json_path', type=str, required=True, help='prediciton result')
    args = parser.parse_args()

    with open(args.json_path) as f:
        data = json.load(f)

    name = np.array(data['name'])
    pred = np.argmax(np.array(data['pred']), axis=1)
    pred_dict = dict(zip(name, pred))

    json_lst = glob.glob('{}/*.json'.format(args.save_path), recursive=True); len(json_lst)

    for json_path in (json_lst):
        base_name = json_path.split('/')[-1].split('.')[0]
        # points, edge_index, _, _ = load_data_v2(json_path,max_edge_lengh=27)
        points, edge_index, _, _ = load_data_v2(json_path,max_edge_lengh=28)
        labels = np.array([pred_dict['{}_{}'.format(base_name, '_'.join(np.array(point, np.str_)))] for point in points])

        with open(json_path) as f:
            data = json.load(f)

        for i in range(len(labels)):
            if labels[i] == 2:
                labels[i] = 0
            data['shapes'][i]['label'] = class_dict[labels[i]]

        with open(json_path, 'w') as f:
            json.dump(data, f)