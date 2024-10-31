import cv2
import copy
import glob
import json
import torch
import numpy as np

from PIL import Image
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset, download_url, Dataset
from scipy.spatial import Voronoi, voronoi_plot_2d, Delaunay
import albumentations as A

def get_training_augmentation():
    train_transform = [
        A.GaussianBlur(),
        A.RandomBrightnessContrast(),
        A.Normalize(),
    ]

    return A.Compose(train_transform)

def get_validation_augmentation():
    test_transform = [
        A.Normalize(),
    ]

    return A.Compose(test_transform)

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

                if np.max(y_sv_norm[ner]) > 0:
                    y_pred[i] = 2
            except:
                None
        else:
            continue
            
    return y_pred


def get_light(img, point, ps=1):
    img_h, img_w = img.shape
    h, w = point
    
    hs = np.clip(h-ps, 0, img_h-2*ps-1)
    ws = np.clip(w-ps, 0, img_w-2*ps-1)
    he = hs + ps*2 + 1
    we = ws + ps*2 + 1
    
    return img[hs:he, ws:we].flatten()


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

        if np.mean(lights[i]) > light_thres:
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


def load_data(json_path):
    data_dict = {
        'Norm': 0,
        'SV': 1,
        'LineSV': 2,
    }
    
    img_path = json_path.replace('.json', '.jpg')
    img = np.array(Image.open(img_path))
    # img = cv2.equalizeHist(img)
    # img = cv2.GaussianBlur(img, (5, 5), 0)

    with open(json_path) as f:
        json_data = json.load(f)
        
    points = np.array([item['points'][0][::-1] for item in json_data['shapes']], np.int32)
    try:
        labels = np.array([data_dict[item['label']] for item in json_data['shapes']], np.uint8)
    except:
        labels = np.array([0 for item in json_data['shapes']], np.uint8)
        
    lights = np.array([get_light(img, point) for point in points])
    edge_index = get_edge_index(points, lights)
    
    return points, edge_index, labels, lights


def load_data_v2(json_path):
    # data_dict = {
    #     'V': 0,
    #     'S': 1,
    #     'Mo': 2,
    # }
    data_dict = {
        'Norm': 0,
        'SV': 1,
        'LineSV': 2,
    }

    img_path = json_path.replace('.json', '.jpg')
    img = np.array(Image.open(img_path))
    # img = cv2.equalizeHist(img)
    # img = cv2.GaussianBlur(img, (5, 5), 0)

    with open(json_path) as f:
        json_data = json.load(f)

    points = np.array([item['points'][0][::-1] for item in json_data['shapes']], np.int32)
    try:
        labels = np.array([data_dict[item['label']] for item in json_data['shapes']], np.uint8)
    except:
        labels = np.array([0 for item in json_data['shapes']], np.uint8)

    lights = np.array([get_light(img, point) for point in points])
    # edge_index = get_edge_index_delaunay(points, max_edge_length=35.5)
    edge_index = get_edge_index_delaunay(points, max_edge_length=28)

    #line 35

    return points, edge_index, labels, lights


def load_data_v2_aug(json_path,aug):
    # data_dict = {
    #     'V': 0,
    #     'S': 1,
    #     'Mo': 2,
    # }
    data_dict = {
        'Norm': 0,
        'SV': 1,
        'LineSV': 2,
    }

    img_path = json_path.replace('.json', '.jpg')
    image = Image.open(img_path)

    # 应用数据增强管道到图像
    img = aug(image=np.array(image))['image']
    # img = np.array(Image.open(img_path))
    # img = cv2.equalizeHist(img)
    # img = cv2.GaussianBlur(img, (5, 5), 0)

    with open(json_path) as f:
        json_data = json.load(f)

    points = np.array([item['points'][0][::-1] for item in json_data['shapes']], np.int32)
    try:
        labels = np.array([data_dict[item['label']] for item in json_data['shapes']], np.uint8)
    except:
        labels = np.array([0 for item in json_data['shapes']], np.uint8)

    lights = np.array([get_light(img, point) for point in points])
    edge_index = get_edge_index_delaunay(points, max_edge_length=35.5)

    #line 35

    return points, edge_index, labels, lights


def load_data_v3(json_path):
    data_dict = {
        'Norm': 0,
        'SV': 1,
        'LineSV': 2,
        'center': 3,
    }

    img_path = json_path.replace('.json', '.jpg')
    img = np.array(Image.open(img_path))
    # img = cv2.equalizeHist(img)
    # img = cv2.GaussianBlur(img, (5, 5), 0)

    with open(json_path) as f:
        json_data = json.load(f)

    points = np.array([item['points'][0][::-1] for item in json_data['shapes']], np.int32)
    try:
        labels = np.array([data_dict[item['label']] for item in json_data['shapes']], np.uint8)
    except:
        labels = np.array([0 for item in json_data['shapes']], np.uint8)

    lights = np.array([get_light(img, point) for point in points])
    edge_index = get_edge_index_delaunay(points, max_edge_length=36)

    return points, edge_index, labels, lights


def load_data_v4(json_path):
    data_dict = {
        'Norm': 0,
        'SV': 1,
        'LineSV': 2,
        'center': 3,
    }

    img_path = json_path.replace('.json', '.jpg')
    img = np.array(Image.open(img_path))
    # img = cv2.equalizeHist(img)
    # img = cv2.GaussianBlur(img, (5, 5), 0)

    with open(json_path) as f:
        json_data = json.load(f)

    points = np.array([item['points'][0][::-1] for item in json_data['shapes']], np.int32)
    try:
        labels = np.array([data_dict[item['label']] for item in json_data['shapes']], np.uint8)
    except:
        labels = np.array([0 for item in json_data['shapes']], np.uint8)

    lights = np.array([get_light(img, point) for point in points])
    edge_index = get_edge_index(points, lights)

    return points, edge_index, labels, lights





def load_data_v5(json_path):
    data_dict = {
        'Norm': 0,
        'SV': 1,
        'LineSV': 2,
        'graph': 3,
        'graphcenter': 4,
    }

    img_path = json_path.replace('.json', '.jpg')
    img = np.array(Image.open(img_path))
    # img = cv2.equalizeHist(img)
    # img = cv2.GaussianBlur(img, (5, 5), 0)

    with open(json_path) as f:
        json_data = json.load(f)

    points = np.array([item['points'][0][::-1] for item in json_data['shapes']], np.int32)
    try:
        labels = np.array([data_dict[item['label']] for item in json_data['shapes']], np.uint8)
    except:
        labels = np.array([0 for item in json_data['shapes']], np.uint8)

    lights = np.array([get_light(img, point) for point in points])
    edge_index = get_edge_index_delaunay(points, max_edge_length=35)

    return points, edge_index, labels, lights



def load_data_v6(json_path):
    data_dict = {
        'Norm': 0,
        'SV': 1,
        'LineSV': 2,
        'graph': 3,
        'graphcenter': 4,
        'vacancy':5
    }

    img_path = json_path.replace('.json', '.jpg')
    img = np.array(Image.open(img_path))
    # img = cv2.equalizeHist(img)
    # img = cv2.GaussianBlur(img, (5, 5), 0)

    with open(json_path) as f:
        json_data = json.load(f)

    points = np.array([item['points'][0][::-1] for item in json_data['shapes']], np.int32)
    try:
        labels = np.array([data_dict[item['label']] for item in json_data['shapes']], np.uint8)
    except:
        labels = np.array([0 for item in json_data['shapes']], np.uint8)

    lights = np.array([get_light(img, point) for point in points])
    edge_index = get_edge_index_delaunay(points, max_edge_length=35)

    return points, edge_index, labels, lights

def find_ner(idx, edge_index):
    ner = list(set(edge_index[1][edge_index[0] == idx].tolist() + edge_index[0][edge_index[1] == idx].tolist()))
    return ner


def get_adj_matrix(n_nodes):
    edges_dic = {}
    if n_nodes in edges_dic:
        edges_dic_b = edges_dic[n_nodes]
        if batch_size in edges_dic_b:
            return edges_dic_b[batch_size]
        else:
            # get edges for a single sample
            rows, cols = [], []
            for batch_idx in range(batch_size):
                for i in range(n_nodes):
                    for j in range(n_nodes):
                        rows.append(i + batch_idx*n_nodes)
                        cols.append(j + batch_idx*n_nodes)

    else:
        edges_dic[n_nodes] = {}
        return get_adj_matrix(n_nodes, batch_size, device)

    edges = [torch.LongTensor(rows).to(device), torch.LongTensor(cols)]
    return edges
    

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




class AtomDataset_v2_aug(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(AtomDataset_v2_aug, self).__init__(root, transform, pre_transform)
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
        json_lst = glob.glob('{}/*.json'.format(self.root), recursive=True)
        print('Json data number: {}.'.format(len(json_lst)))

        data_lst = []
        # for  i in range(10):
        for json_path in json_lst:
            base_name = json_path.split('/')[-1].split('.')[0]
            aug = get_training_augmentation() if json_path.split('/')[-2] == 'train' else get_validation_augmentation()
            points, edge_index, labels, lights = load_data_v2_aug(json_path,aug)

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