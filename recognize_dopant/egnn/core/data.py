import cv2
import copy
import glob
import json
import torch
import numpy as np

from PIL import Image
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset, download_url, Dataset

# from utils.cycles import get_graph, find_cycles

from scipy.spatial import Voronoi, voronoi_plot_2d, Delaunay

def get_light(img, point, ps=1):
    img_h, img_w = img.shape
    h, w = point
    
    hs = np.clip(h-ps, 0, img_h-2*ps-1)
    ws = np.clip(w-ps, 0, img_w-2*ps-1)
    he = hs + ps*2 + 1
    we = ws + ps*2 + 1
    
    return img[hs:he, ws:we].flatten()


def get_edge_index(points, max_ner=3, dis_thres=30, light_thres=120):
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

        s += [i] * len(ner_final_idxs)
        t += ner_final_idxs
        connected_id += [i]

    edge_index = np.array([s, t])
    
    edge_str = []
    edge_index = np.sort(edge_index, axis=0)

    for item in edge_index.transpose():
        edge_str += ['{}_{}'.format(item[0], item[1])]

    edge_str = list(set(edge_str))

    s = [int(item.split('_')[0]) for item in edge_str]
    e = [int(item.split('_')[1]) for item in edge_str]

    edge_index = np.array([s+e, e+s])
    
    return edge_index


def get_edge_index_delaunay(points, max_edge_length=35):
    tri = Delaunay(points)

    edge_index = set()
    for simplex in tri.simplices:
        for i in range(3):
            for j in range(i + 1, 3):
                edge = tuple(sorted([simplex[i], simplex[j]]))
                edge_length = np.linalg.norm(points[edge[0]] - points[edge[1]])
                if edge_length <= max_edge_length:
                    edge_index.add(edge)

    edge_index = np.array(list(edge_index)).T

    edge_str = []
    edge_index = np.sort(edge_index, axis=0)

    for item in edge_index.transpose():
        edge_str += ['{}_{}'.format(item[0], item[1])]

    edge_str = list(set(edge_str))

    s = [int(item.split('_')[0]) for item in edge_str]
    e = [int(item.split('_')[1]) for item in edge_str]

    edge_index = np.array([s + e, e + s])

    return edge_index

def load_data_v2(json_path):
    data_dict = {
        'Norm': 0,
        'atom': 0,
        'cz': 1,
    }

    img_path = json_path.replace('.json', '.jpg')
    img = np.array(Image.open(img_path))

    with open(json_path) as f:
        json_data = json.load(f)

    points = np.array([item['points'][0][::-1] for item in json_data['shapes']], np.int32)
    lights = np.array([get_light(img, point) for point in points])
    labels = np.array([data_dict[item['label']] for item in json_data['shapes']], np.int32)

    edge_index = get_edge_index_delaunay(points, max_edge_length=28)

    return points, edge_index, labels, lights

def load_data(json_path):
    data_dict = {
        'atom': 0,
        '掺杂': 1,
    }
    
    img_path = json_path.replace('.json', '.jpg')
    img = np.array(Image.open(img_path))

    with open(json_path) as f:
        json_data = json.load(f)
        
    points = np.array([item['points'][0][::-1] for item in json_data['shapes']], np.int32)
    lights = np.array([get_light(img, point) for point in points])
    labels = np.array([data_dict[item['label']] for item in json_data['shapes']], np.int32)
    edge_index = get_edge_index(points)
    
    return points, edge_index, labels, lights


def find_ner(idx, edge_index):
    ner = list(set(edge_index[1][edge_index[0] == idx].tolist() + edge_index[0][edge_index[1] == idx].tolist()))
    return ner


class AtomDataset_v2(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(AtomDataset_v2, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['atom.dataset']

    def download(self):
        pass

    def process(self):
        json_lst = glob.glob('{}/raw/*.json'.format(self.root), recursive=True)
        print('Json data number: {}.'.format(len(json_lst)))

        data_lst = []
        for json_path in json_lst:
            base_name = json_path.split('/')[-1].split('.')[0]
            points, edge_index, labels, lights = load_data_v2(json_path)

            for idx, center in enumerate(points):
                ner1 = find_ner(idx, edge_index)
                ner2 = []
                if len(ner1) != 0:
                    ner2 = [find_ner(item, edge_index) for item in ner1]
                    ner2 = [item2 for item1 in ner2 for item2 in item1]
                    ner2 = list(set(ner2))
                    if idx in ner2:
                        ner2.remove(idx)

                ner = [idx] + list(set(ner1 + ner2))
                atom_nums = len(ner)

                sample_light = torch.FloatTensor(lights[ner]) / 255.
                sample_pos = torch.FloatTensor((points[ner] - np.mean(points[ner], axis=0)) / 12.)
                sample_label = torch.tensor([labels[idx]] + [0] * (atom_nums - 1), dtype=torch.int64)
                sample_mask = torch.tensor([1] + [0] * (atom_nums - 1), dtype=torch.bool)
                sample_edge_index = torch.LongTensor(
                    np.array([(i, j) for i in range(atom_nums) for j in range(atom_nums)]).transpose())

                sample_data = Data(
                    pos=sample_pos,
                    light=sample_light,
                    label=sample_label,
                    mask=sample_mask,
                    edge_index=sample_edge_index,
                    name='{}_{}_{}'.format(base_name, center[0], center[1])
                )

                data_lst += [sample_data]

        data, slices = self.collate(data_lst)
        torch.save((data, slices), self.processed_paths[0])

class AtomDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(AtomDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
    
    @property
    def raw_file_names(self):
        return []
    
    @property
    def processed_file_names(self):
        return ['atom.dataset']
    
    def download(self):
        pass
    
    def process(self):
        json_lst = glob.glob('{}/raw/*.json'.format(self.root), recursive=True)
        print('Json data number: {}.'.format(len(json_lst)))
        
        data_lst = []
        for json_path in json_lst:
            base_name = json_path.split('/')[-1].split('.')[0]
            points, edge_index, labels, lights = load_data(json_path)

            for idx, center in enumerate(points):
                ner1 = find_ner(idx, edge_index)
                ner2 = []
                if len(ner1) != 0:
                    ner2 = [find_ner(item, edge_index) for item in ner1]
                    ner2 = [item2 for item1 in ner2 for item2 in item1]
                    ner2 = list(set(ner2))
                    if idx in ner2:
                        ner2.remove(idx)
                        
                ner = [idx] + list(set(ner1 + ner2))
                atom_nums = len(ner)

                sample_light = torch.FloatTensor(lights[ner]) / 255.
                sample_pos = torch.FloatTensor((points[ner] - np.mean(points[ner], axis=0)) / 12.)
                sample_label = torch.tensor([labels[idx]] + [0]*(atom_nums - 1), dtype=torch.int64)
                sample_mask = torch.tensor([1] + [0]*(atom_nums - 1), dtype=torch.bool)
                sample_edge_index = torch.LongTensor(np.array([(i, j) for i in range(atom_nums) for j in range(atom_nums)]).transpose())

                sample_data = Data(
                    pos = sample_pos,
                    light = sample_light,
                    label = sample_label,
                    mask = sample_mask,
                    edge_index = sample_edge_index,
                    name = '{}_{}_{}'.format(base_name, center[0], center[1])
                )

                data_lst += [sample_data]

        data, slices = self.collate(data_lst)
        torch.save((data, slices), self.processed_paths[0])  