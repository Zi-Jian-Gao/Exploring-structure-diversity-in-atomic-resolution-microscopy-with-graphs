import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.feature import peak_local_max


def norm_0_1(mask):
    x_min = np.min(mask)
    x_max = np.max(mask)
    
    new_mask = (mask-x_min) / (x_max-x_min)
    
    return new_mask
    

def get_dotsmap(den_map, min_dis, thres):
    
    if np.max(den_map) < thres:
        return []
    
    den_map = norm_0_1(den_map)
    
    x_y = peak_local_max(
        den_map, 
        min_distance = min_dis,
        threshold_abs = thres,
    )
    
    dots_map = np.zeros(den_map.shape)
    dots_map[x_y[:, 0].tolist(), x_y[:, 1].tolist()] = 1
    
    return dots_map
    