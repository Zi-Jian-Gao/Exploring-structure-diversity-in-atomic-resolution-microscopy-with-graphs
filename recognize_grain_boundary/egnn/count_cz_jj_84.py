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
        count_8_atoms = 0
        count_4_atoms = 0
        # 遍历 points 数组中的每个原子坐标
        for point in points:
            # 重置标志变量
            continue_outer_loop = True

            # 检查8个原子的列表是否存在
            if 8 in point_lst_dict:
                # 如果存在，检查坐标是否属于8个原子的列表
                for atom_point_lst in point_lst_dict[8]:
                    if any(np.allclose(point, atom_point) for atom_point in atom_point_lst):
                        count_8_atoms += 1
                        continue_outer_loop = False  # 找到匹配后，设置标志变量为False，以退出外层循环
                        break  # 退出当前的内层循环

            # 如果8的列表不存在或者坐标不属于8个原子的列表，检查6个原子的列表
            # 这里省略了6个原子的检查，因为题目中没有提供6个原子的列表

            # 如果8和6的列表都不存在或者坐标不属于它们，检查4个原子的列表
            if 4 in point_lst_dict and continue_outer_loop:
                for atom_point_lst in point_lst_dict[4]:
                    if any(np.allclose(point, atom_point) for atom_point in atom_point_lst):
                        count_4_atoms += 1
                        continue_outer_loop = False  # 找到匹配后，设置标志变量为False，以退出外层循环
                        break  # 退出当前的内层循环

            # 根据标志变量决定是否继续外层循环
            # if not continue_outer_loop:
            #     break

        # 打印结果
        print(f"参杂在jj上的原子数量: {cz_on_jj}")
        print(f"属于8个原子数量的list的原子数量: {count_8_atoms}")
        print(f"属于4个原子数量的list的原子数量: {count_4_atoms}")

    return cz_on_jj,count_8_atoms, count_4_atoms

import argparse
if __name__ == '__main__':
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description='Process some images.')
    # 添加 --path 参数
    parser.add_argument('--save_path', type=str, required=True, help='Path to the images directory')

    # 解析命令行参数
    args = parser.parse_args()

    json_lst = glob.glob('{}/*.json'.format(args.save_path), recursive=True); len(json_lst)
    total_cz_on_jj = 0
    total_count_8_atoms = 0
    total_count_4_atoms = 0
    for json_path in (json_lst):
        # print(json_path)
        base_name = json_path.split('/')[-1].split('.')[0]

        points, edge_index, labels, lights = load_data_jj(json_path,max_edge_lengh=28)

        from collections import defaultdict

        data_lst = []
        point_lst_dict = defaultdict(list)  # 使用字典来存储不同原子数量的point_lst

        graph = get_graph(edge_index)

        for atom_nums in [4, 5, 7, 8]:
            cycles = np.array(list(find_cycles(graph, atom_nums)))

            idx = np.array([np.sum([(item[0] in cycle) & (item[1] in cycle) for item in edge_index.T]) for cycle in
                            cycles]) == 2 * atom_nums

            cycles = cycles[idx]

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

                data_lst.append(sample_data)  # 使用append而不是+=，以避免引用问题
                point_lst_dict[atom_nums].append(points[atoms_idx])


                # img_path = json_path.replace('json', 'jpg')
        cz_json_path = json_path.replace('gb', 'dophant_atom')

        cz_on_jj,count_8_atoms, count_4_atoms = count(cz_json_path, labels,point_lst_dict)
        total_cz_on_jj += cz_on_jj
        total_count_8_atoms += count_8_atoms
        total_count_4_atoms += count_4_atoms

    print(f"所有参杂在jj上的原子数量: {total_cz_on_jj}")
    print(f"所有属于8个原子数量的list的原子数量: {total_count_8_atoms}")
    print(f"所有属于4个原子数量的list的原子数量: {total_count_4_atoms}")
    # print(f"属于4个原子数量的list的原子数量: {count_4_atoms}")


