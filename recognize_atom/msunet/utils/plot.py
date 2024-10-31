import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def plot4(json_path, i):
    with open(json_path, 'r') as f:
        result = json.load(f)
        
    imgs = np.array(result['img_path'])
    preds = np.array(result['pred'])
    labels = np.array(result['label'])
    
    plt.figure(figsize=(8, 2))
    plt.subplot(1, 4, 1)
    img = Image.open(imgs[i])
    plt.imshow(img)
    plt.axis('off')
    plt.subplot(1, 4, 2)
    plt.imshow(labels[i])
    plt.axis('off')
    plt.subplot(1, 4, 3)
    plt.imshow(preds[i])
    plt.axis('off')
    plt.subplot(1, 4, 4)
    dmap = get_dotsmap(preds[i])
    h, w = np.where(dmap != 0)
    plt.imshow(img, cmap='gray')
    plt.scatter(w, h, c='r')
    plt.axis('off')