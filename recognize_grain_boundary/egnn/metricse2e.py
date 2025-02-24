import os
import cv2
import glob
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from PIL import Image
from utils.e2e_metrics import get_metrics
from core.data import get_y_3
from core.data import load_data_vor, find_cycles, get_graph
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
import argparse
from torch_geometric.data import Data

class_dict = {
    0: 'JJ',
    1: 'Norm',
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

        print(json_path)
        base_name = json_path.split('/')[-1].split('.')[0]

        points, edge_index, lights = load_data_vor(json_path)

        data_lst = []
        point_lst = []
        graph = get_graph(edge_index)
        # for atom_nums in [3, 4, 5, 6, 7, 8]:
        for atom_nums in [4, 5, 6, 7, 8]:

            cycles = np.array(list(find_cycles(graph, atom_nums)))

            idx = np.array([np.sum([(item[0] in cycle) & (item[1] in cycle) for item in edge_index.T]) for cycle in
                            cycles]) == 2 * atom_nums

            cycles = cycles[idx]
            # print(cycles.shape)
            for idx, atoms_idx in enumerate(cycles):
                sample_light = torch.FloatTensor(lights[atoms_idx]) / 255.
                sample_pos = torch.FloatTensor((points[atoms_idx] - np.mean(points[atoms_idx], axis=0)) / 12.)
                sample_label = torch.tensor([int(atom_nums == 6)], dtype=torch.int64)
                sample_edge_index = torch.LongTensor(
                    np.array([(i, j) for i in range(atom_nums) for j in range(atom_nums)]).transpose())

                sample_data = Data(
                    pos=sample_pos,
                    light=sample_light,
                    label=sample_label,
                    edge_index=sample_edge_index,
                    name='{}_{}_{}'.format(base_name, atom_nums, idx)
                )

                data_lst += [sample_data]
                point_lst.append(points[atoms_idx])

        pred_indx = [i for i, x in enumerate(name) if x.startswith(f"{base_name}_")]
        cur_pred = pred[pred_indx]
        with open(json_path) as f:
            data = json.load(f)
        labels = np.ones(len(data['shapes']))


        indexes_of_false = [index for index, value in enumerate(cur_pred) if value == 0]

        for idx in indexes_of_false:
            for point in point_lst[idx]:
                label_idx = [index for index, value in enumerate(points) if all(v == p for v, p in zip(value, point))]
                labels[label_idx[0]] = 0

        for i in range(len(labels)):
            data['shapes'][i]['label'] = class_dict[int(labels[i])]

        with open(json_path, 'w') as f:
            json.dump(data, f)
