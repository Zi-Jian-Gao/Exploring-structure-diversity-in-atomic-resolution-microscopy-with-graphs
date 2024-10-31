import cv2
import copy
import glob
import json
import torch
import numpy as np

from PIL import Image
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset, download_url, Dataset


def get_y_3(y, edge_index):
    conn_matrix = np.zeros((len(y), len(y)))

    for s, e in edge_index.transpose():
        conn_matrix[s, e] = 1
        conn_matrix[e, s] = 1

    y_sv_norm = np.array(y != 0, np.uint8)
    y_pred = np.array(y != 0, np.uint8)

    for i in range(len(y_pred)):
        if y_pred[i] != 0:
            try:
                n1 = np.where(conn_matrix[i] != 0)[0]
                n2 = np.concatenate([np.where(conn_matrix[item] != 0)[0] for item in n1])
                ner = np.concatenate([n1, n2])
                ner = set(ner)
                ner.remove(i)
                ner = list(ner)

                if np.max(y_sv_norm[ner]) == 0:
                    y_pred[i] = 2
            except:
                None
        else:
            continue
            
    return y_pred


def get_light(img, point):
    h, w = point
    
    return img[h, w]


def get_edge_index(points, lights, max_ner=3, dis_thres=35, light_thres=120):
    s = []
    t = []
    connected_id = []

    for i, p in enumerate(points):
        p = p.reshape(1, -1)
        xysub = np.abs(p - points)
        diss = np.sqrt(xysub[:, 0] ** 2 + xysub[:, 1] ** 2)

        ner_idxs = np.argsort(diss)[1: 1+max_ner].tolist()

        for j, item in enumerate(ner_idxs):
            if item in connected_id:
                ner_idxs.pop(j)

        ner_idxs = np.array(ner_idxs)
        ner_diss = diss[ner_idxs]
        ner_final_idxs = ner_idxs[ner_diss < dis_thres].tolist()

        if lights[i] > light_thres:
            s += [i] * len(ner_final_idxs)
            t += ner_final_idxs
            connected_id += [i]

    edge_index = np.array([s, t])
    
    return edge_index


def load_data(json_path):
    data_dict = {
        'Norm': 0,
        'LineSV': 1,
        'SV': 2
    }
    
    img_path = json_path.replace('.json', '.jpg')
    img = np.array(Image.open(img_path))

    with open(json_path) as f:
        json_data = json.load(f)
        
    points = np.array([item['points'][0][::-1] for item in json_data['shapes']], np.int32)
    labels = np.array([data_dict[item['label']] for item in json_data['shapes']], np.uint8)
    lights = np.array([get_light(img, point) for point in points])
    edge_index = get_edge_index(points, lights)
    
    return points, edge_index, labels, lights
