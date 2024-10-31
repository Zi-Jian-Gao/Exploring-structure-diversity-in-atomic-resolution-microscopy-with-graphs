import cv2
import copy
import glob
import json
import torch
import numpy as np

from PIL import Image
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset, download_url, Dataset

def get_light(img, point, half_ps=3):
    
    h, w = point
    
    hs = h - half_ps
    he = hs + half_ps*2 + 1
    ws = w - half_ps
    we = ws + half_ps*2 + 1
    
    return np.median(img[hs:he, ws:we].flatten())
    

def load_data(json_path):
    data_dict = {
        'Norm': 0,
        'LineSV': 1,
        'SV': 2
    }
    
    img_path = json_path.replace('.json', '.jpg')
    img = np.array(Image.open(img_path))
    img = cv2.GaussianBlur(img, (5, 5), 0)
    
    with open(json_path) as f:
        json_data = json.load(f)
        
    points = np.array([item['points'][0][::-1] for item in json_data['shapes']], np.int32)
    labels = np.array([data_dict[item['label']] for item in json_data['shapes']], np.uint8)
    lights = np.array([get_light(img, point) for point in points])
    
    return img, points, labels, lights
    

def get_patch(img, points, patch_size):
    h, w = img.shape
    padding_size = int(patch_size/2)
    
    img_pad = np.zeros((h+padding_size*2, w+padding_size*2))
    img_pad[padding_size:padding_size+h, padding_size:padding_size+w] = img
    cpoints = copy.deepcopy(points) + padding_size
    
    patch = []
    for point in cpoints:
        ph, pw = point
        phs = ph - padding_size
        pws = pw - padding_size
        pp = img_pad[phs:phs+patch_size, pws:pws+patch_size]
        pp = pp[np.newaxis, np.newaxis, :, :]
        pp = np.repeat(pp, 3, axis=1)
        patch += [pp]

    patch = np.concatenate(patch)
    
    return patch
    

def get_con(json_path, patch_size, dis_thres=35, max_ner=3, light_thres=120):
    img, points, labels, lights = load_data(json_path)
    patch = get_patch(img, points, patch_size=patch_size)
    
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
        
    edge_attr = np.sqrt(((points[s] - points[t]) ** 2).sum(1))  
    edge_index = torch.LongTensor(np.array([s, t]))
    
    # features
    x = torch.FloatTensor(patch)
    y = torch.LongTensor(labels != 0)
    
    return x, y, edge_index, edge_attr, points, img, labels
    

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
        
        data_list = []
        for json_path in json_lst:
            name = json_path.split('/')[-1].split('.')[0]
            x, y, edge_index, _, points, _, _ = get_con(json_path, patch_size=48)
            data = Data(x=x, y=y, edge_index=edge_index, points=points, name=name)
            data_list.append(data)
        
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])