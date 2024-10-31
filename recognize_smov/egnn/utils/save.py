import os
import json
import torch
import numpy as np

def save_results(logdir, predictions, save_name):
    preds = torch.squeeze(torch.concat([item[0] for item in predictions])).numpy().tolist()
    labels = torch.squeeze(torch.concat([item[1] for item in predictions])).numpy().tolist()
    name = [item[2] for item in predictions]
    
    results ={
        'name': name,
        'pred': preds,
        'label': labels,
    }
    
    results_json = json.dumps(results)
    with open(os.path.join(logdir, '{}.json'.format(save_name)), 'w+') as f:
        f.write(results_json)
        