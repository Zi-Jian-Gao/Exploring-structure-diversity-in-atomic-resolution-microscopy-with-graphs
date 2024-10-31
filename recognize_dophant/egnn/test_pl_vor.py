import os
import json
import glob
import torch
import random
torch.set_float32_matmul_precision('high')

import numpy as np
from torch import nn
from torch.utils.data.sampler import *
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import pytorch_lightning as pl

from core.model import PL_EGNN
from core.data import *
from core.metrics import *
from core.aug import *
from torch_geometric.data.lightning import LightningDataset
from torch.utils.data import ConcatDataset
from utils.save import save_results
import argparse
# constants



DATASETS = '0'
GPUS = 1
BS = 128
NW = 1
WD = 5e-4
LR = 0.01
RF = 0.9
EPOCHS = 1000
IN_CHANNELS = 3
SAVE_TOP_K = 1
EARLY_STOP = 16
EVERY_N_EPOCHES = 2


# pytorch lightning module

class PLModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = PL_EGNN()
        self.loss = nn.CrossEntropyLoss(torch.FloatTensor([1., 4.]))

    def forward(self, x):
        out = self.model(x.x, x.edge_index, x.batch)
        return out

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=LR, weight_decay=WD)
        scheduler = ReduceLROnPlateau(optimizer, factor=RF, mode='max', patience=2, min_lr=0, verbose=True)
        return {
           'optimizer': optimizer,
           'lr_scheduler': scheduler,
           'monitor': 'val_acc'
        }

    def training_step(self, train_batch, batch_idx):
        pos, light, y, edge_index, batch, mask = (
            train_batch.pos,
            train_batch.light,
            train_batch.label,
            train_batch.edge_index,
            train_batch.batch,
            train_batch.mask,
        )
        
        pd = self.model(light, pos, edge_index)
        loss = self.loss(pd[mask], y[mask])
        train_acc = acc(pd[mask], y[mask])
        self.log('train_loss', loss, batch_size=1)
        self.log('train_acc', train_acc, on_epoch=True, prog_bar=True, logger=True, batch_size=1)
        
        return loss

    def validation_step(self, val_batch, batch_idx):        
        pos, light, y, edge_index, batch, mask = (
            val_batch.pos,
            val_batch.light,
            val_batch.label,
            val_batch.edge_index,
            val_batch.batch,
            val_batch.mask,
        )
        
        pd = self.model(light, pos, edge_index)
        loss = self.loss(pd[mask], y[mask])
        val_acc = acc(pd[mask], y[mask])
        self.log('val_loss', loss, batch_size=1)
        self.log('val_acc', val_acc, on_epoch=True, prog_bar=True, logger=True, batch_size=1)
        

    def predict_step(self, batch, batch_idx):
        pos, light, y, edge_index, batch, mask, name = (
            batch.pos,
            batch.light,
            batch.label,
            batch.edge_index,
            batch.batch,
            batch.mask,
            batch.name
        )   
        
        pd = self.model(light, pos, edge_index)
        return pd[mask], y[mask], name

# main

if __name__ == '__main__':
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description='Process some images.')
    # 添加 --path 参数
    parser.add_argument('--path', type=str, required=True, help='Path to the images directory')

    # 解析命令行参数
    args = parser.parse_args()

    test_dataset = AtomDataset_v2(root=args.path)

    datamodule = LightningDataset(
        test_dataset,
        test_dataset,
        test_dataset,
        batch_size=BS,
        num_workers=NW,
    )

    model = PLModel()
    
    logger = TensorBoardLogger(
        name = DATASETS,
        save_dir = 'logs',
    )

    checkpoint_callback = ModelCheckpoint(
        every_n_epochs = EVERY_N_EPOCHES,
        save_top_k = SAVE_TOP_K,
        monitor = 'val_acc',
        mode = 'max',
        save_last = True,
        filename = '{epoch}-{val_loss:.2f}-{val_acc:.2f}'
    )

    earlystop_callback = EarlyStopping(
        monitor = "val_acc", 
        mode = "max",
        min_delta = 0.00, 
        patience = EARLY_STOP,
    )

    # training
    trainer = pl.Trainer(
        log_every_n_steps=1,
        devices = GPUS,
        max_epochs = EPOCHS,
        logger = logger,
        callbacks = [earlystop_callback, checkpoint_callback],
    )


    predictions = trainer.predict(
        model, 
        datamodule.test_dataset,
        ckpt_path = './logs/model/checkpoints/last.ckpt',
   )
    save_results(trainer.log_dir, predictions, 'test')
