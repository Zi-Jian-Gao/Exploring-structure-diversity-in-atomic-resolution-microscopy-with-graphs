import os
import json
import glob
import torch
torch.set_float32_matmul_precision('high')

import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import *
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import pytorch_lightning as pl

from core.model import *
from core.data import *
from core.metrics import *
import argparse
# constants

DATASETS = '0'
GPUS = 1
SIGMA = 3
BS = 32
RF = 0.9
NW = 8
WD = 1e-5
LR = 3e-4
DIM = 256
# SLIDE_DIM = 2048
patch_size = 256
roi_size = 128
# roi_size = 128+64
# patch_size = 512
# roi_size = 256
EPOCHS = 1000
IN_CHANNELS = 3
SAVE_TOP_K = -1
EARLY_STOP = 5
EVERY_N_EPOCHS = 1

# IMAGE_PATH2 = '/home/gao/下载/process/test/'
WEIGHTS = torch.FloatTensor([1./8, 1./4, 1./2, 1.])

# pytorch lightning module

class FCRN(pl.LightningModule):
    def __init__(self, in_channels):
        super().__init__()
        self.fcrn = C_FCRN_Aux(in_channels)
        self.loss = MyLoss(WEIGHTS)

    def forward(self, x):
        out = self.fcrn(x)
        return out

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=LR, weight_decay=WD)
        scheduler = ReduceLROnPlateau(optimizer, factor=RF, mode='max', patience=2, min_lr=0, verbose=True)
        return {
           'optimizer': optimizer,
           'lr_scheduler': scheduler,
           'monitor': 'val_dice'
        }

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        pd = self.fcrn(x)
        loss = self.loss(pd, y)
        train_iou = iou(pd, y)
        train_dice = dice(pd, y)
        self.log('train_loss', loss)
        self.log('train_iou', train_iou, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_dice', train_dice, on_epoch=True, prog_bar=True, logger=True)
        
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        pd = self.fcrn(x)
        loss = self.loss(pd, y)
        val_iou = iou(pd, y)
        val_dice = dice(pd, y)
        self.log('val_loss', loss)
        self.log('val_iou', val_iou, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_dice', val_dice, on_epoch=True, prog_bar=True, logger=True)
        
    def predict_step(self, batch, batch_idx):
        x, lbl = batch
        x = torch.chunk(x[0], chunks=4, dim=0)
        pred = torch.concat([self.fcrn(item)[-1] for item in x])
        
        return pred.squeeze(), lbl

# main

# IMAGE_PATH = '../../img'

if __name__ == '__main__':
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description='Process some images.')
    # 添加 --path 参数
    parser.add_argument('--path', type=str, required=True, help='Path to the images directory')

    # 解析命令行参数
    args = parser.parse_args()

    test_img_list = np.array(glob.glob(f'{args.path}/*.jpg')).tolist()


    test_dataset = MyDatasetSlide_test_uneuqal(test_img_list, ps=patch_size,roi=roi_size)

    test_loader = DataLoader(
        dataset = test_dataset,
        batch_size = 1,
        shuffle = False,
        num_workers = 1,
    )

    model = FCRN(IN_CHANNELS)

    
    logger = TensorBoardLogger(
        name = DATASETS,
        save_dir = 'logs',
    )

    checkpoint_callback = ModelCheckpoint(
        every_n_epochs = EVERY_N_EPOCHS,
        save_top_k = SAVE_TOP_K,
        monitor = 'val_dice',
        mode = 'max',
        save_last = True,
        filename = '{epoch}-{val_loss:.2f}-{val_dice:.2f}'
    )

    earlystop_callback = EarlyStopping(
        monitor = "val_dice", 
        mode = "max",
        min_delta = 0.00, 
        patience = EARLY_STOP,
    )
    
    # training
    trainer = pl.Trainer(
        accelerator = 'gpu',
        devices = GPUS,
        max_epochs = EPOCHS,
        logger = logger,
        callbacks = [checkpoint_callback, earlystop_callback],
    )

    # inference
    predictions = trainer.predict(
        model = model,
        dataloaders = test_loader,
        ckpt_path = './logs/model/checkpoints/last.ckpt'
    )

    preds = np.concatenate([test_dataset.spliter.recover(item[0], item[1].shape[1], item[1].shape[2], ps=patch_size,roi=roi_size)[np.newaxis, :, :] for item in predictions]).tolist()
    labels = torch.squeeze(torch.concat([item[1] for item in predictions])).numpy().tolist()

    results ={
        'img_path': test_img_list,
        'pred': preds,
        'label': labels,
    }

    results_json = json.dumps(results)
    with open(os.path.join(trainer.log_dir, 'test.json'), 'w+') as f:
        f.write(results_json)
