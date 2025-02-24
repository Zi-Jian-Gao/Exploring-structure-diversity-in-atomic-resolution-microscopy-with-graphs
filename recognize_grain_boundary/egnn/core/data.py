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
# from cycles import find_cycles,get_graph
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


def find_cycles(graph, length):
    def dfs(node, visited, path):
        if len(path) == length:
            # Check if it forms a cycle
            if path[0] in graph[path[-1]]:
                sorted_path = tuple(sorted(path))
                if sorted_path not in cycles:
                    cycles.add(sorted_path)
            return

        if node not in graph:
            visited[node] = False
            return


        visited[node] = True
        for neighbor in graph[node]:
            if  neighbor == len(visited):
                return
            if not visited[neighbor]:
                dfs(neighbor, visited, path + [neighbor])
        visited[node] = False

    cycles = set()
    num_nodes = len(graph)
    visited = [False] * num_nodes

    for node in range(num_nodes):
        # if node==0:
        #     print('1')
        dfs(node, visited, [node])

    return cycles


def get_graph(edge_index):
    graph = {}
    for edge in edge_index.T:
        u, v = edge
        if u not in graph:
            graph[u] = []
        if v not in graph:
            graph[v] = []
        graph[u].append(v)
        graph[v].append(u)

    return graph

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
    
    return points, edge_index, lights


def load_data_cz(json_path,max_edge_lengh):
    data_dict = {
        'atom': 3,
        'cz': 1,
    }

    img_path = json_path.replace('.json', '.jpg')
    img = np.array(Image.open(img_path))

    with open(json_path) as f:
        json_data = json.load(f)

    points = np.array([item['points'][0][::-1] for item in json_data['shapes']], np.int32)
    try:
        labels = np.array([data_dict[item['label']] for item in json_data['shapes']], np.uint8)
    except:
        labels = np.array([0 for item in json_data['shapes']], np.uint8)

    lights = np.array([get_light(img, point) for point in points])
    edge_index = get_edge_index_delaunay(points, max_edge_length=max_edge_lengh)

    return points, edge_index, labels, lights


def load_data_kw(json_path,max_edge_lengh):
    data_dict = {
        'S': 1,
        'Mo': 2,
        'V': 3,
    }

    img_path = json_path.replace('.json', '.jpg')
    img = np.array(Image.open(img_path))

    with open(json_path) as f:
        json_data = json.load(f)

    points = np.array([item['points'][0][::-1] for item in json_data['shapes']], np.int32)
    try:
        labels = np.array([data_dict[item['label']] for item in json_data['shapes']], np.uint8)
    except:
        labels = np.array([0 for item in json_data['shapes']], np.uint8)

    lights = np.array([get_light(img, point) for point in points])
    edge_index = get_edge_index_delaunay(points, max_edge_length=max_edge_lengh)

    return points, edge_index, labels, lights

def load_data_jj(json_path,max_edge_lengh):
    data_dict = {
        'Norm': 1,
        'JJ': 2,
    }

    img_path = json_path.replace('.json', '.jpg')
    img = np.array(Image.open(img_path))

    with open(json_path) as f:
        json_data = json.load(f)

    points = np.array([item['points'][0][::-1] for item in json_data['shapes']], np.int32)
    try:
        labels = np.array([data_dict[item['label']] for item in json_data['shapes']], np.uint8)
    except:
        labels = np.array([0 for item in json_data['shapes']], np.uint8)

    lights = np.array([get_light(img, point) for point in points])
    edge_index = get_edge_index_delaunay(points, max_edge_length=max_edge_lengh)

    return points, edge_index, labels, lights

def load_data_vor(json_path):
    data_dict = {
        '1': 0,
        '2': 1,
    }
    max_edge_length = 28

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

    # # 挑选出 labels 中值为 0 的索引
    # indices_to_remove = np.where(labels == 0)[0]
    # points = np.delete(points, indices_to_remove, axis=0)

    lights = np.array([get_light(img, point) for point in points])
    edge_index = get_edge_index_delaunay(points, max_edge_length= max_edge_length )

    return points, edge_index, lights


def load_data_vor_aug(json_path,aug):
    data_dict = {
        '1': 0,
        '2': 1,
    }
    max_edge_length = 28

    img_path = json_path.replace('.json', '.jpg')

    image = Image.open(img_path)

    # 应用数据增强管道到图像
    img = aug(image=np.array(image))['image']

    # 将增强后的图像转换回PIL图像格式
    # augmented_pil_image = Image.fromarray(augmented_image)

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

    # # 挑选出 labels 中值为 0 的索引
    # indices_to_remove = np.where(labels == 0)[0]
    # points = np.delete(points, indices_to_remove, axis=0)

    lights = np.array([get_light(img, point) for point in points])
    edge_index = get_edge_index_delaunay(points, max_edge_length= max_edge_length )

    return points, edge_index, lights

def find_ner(idx, edge_index):
    ner = list(set(edge_index[1][edge_index[0] == idx].tolist() + edge_index[0][edge_index[1] == idx].tolist()))
    return ner


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
            points, edge_index, lights = load_data(json_path)

            for atom_nums in [5, 6, 7]:
                graph = get_graph(edge_index)
                cycles = np.array(list(find_cycles(graph, atom_nums)))

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

        data, slices = self.collate(data_lst)
        torch.save((data, slices), self.processed_paths[0])


class AtomDataset_vor(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, data_type=None):
        super(AtomDataset_vor, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        # self.aug = None
        # self.aug = get_training_augmentation() if data_type == 'train' else get_validation_augmentation()

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
        for json_path in json_lst:
            print(json_path)
            base_name = json_path.split('/')[-1].split('.')[0]
            points, edge_index, lights = load_data_vor(json_path)

            graph = get_graph(edge_index)
            for atom_nums in [3, 4, 5, 6, 7, 8]:

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

                    data_lst += [sample_data]

        data, slices = self.collate(data_lst)
        torch.save((data, slices), self.processed_paths[0])


class AtomDataset_vor_aug(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, data_type=None):
        super(AtomDataset_vor_aug, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        # self.aug = None


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
        for json_path in json_lst:
            print(json_path)
            aug = get_training_augmentation() if json_path.split('/')[-2] == 'train' else get_validation_augmentation()

            base_name = json_path.split('/')[-1].split('.')[0]
            points, edge_index, lights = load_data_vor_aug(json_path,aug)

            graph = get_graph(edge_index)
            # for atom_nums in [3, 4, 5, 6, 7, 8]:
            for atom_nums in [4, 5, 6, 7, 8]:

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

                    data_lst += [sample_data]

        data, slices = self.collate(data_lst)
        torch.save((data, slices), self.processed_paths[0])