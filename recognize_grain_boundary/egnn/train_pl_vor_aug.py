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

# constants

DATASETS = '0'
GPUS = 1
BS = 32
NW = 1
WD = 5e-4
LR = 0.01
RF = 0.9
EPOCHS = 1000
IN_CHANNELS = 3
SAVE_TOP_K = 5
EARLY_STOP = 5
EVERY_N_EPOCHES = 2
DATA_PATH = '/home/gao/mouclear/cc/data/e2e_jj_point'

# pytorch lightning module

class PLModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = PL_EGNN()
        # self.loss = nn.CrossEntropyLoss(torch.FloatTensor([1., 2., 2.]))
        self.loss = nn.CrossEntropyLoss(torch.FloatTensor([1., 2.]))

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
        pos, light, y, edge_index, batch = (
            train_batch.pos,
            train_batch.light,
            train_batch.label,
            train_batch.edge_index,
            train_batch.batch,
        )
        
        pd = self.model(light, pos, edge_index,train_batch.batch)
        loss = self.loss(pd, y)
        train_acc = acc(pd, y)
        self.log('train_loss', loss, batch_size=1)
        self.log('train_acc', train_acc, on_epoch=True, prog_bar=True, logger=True, batch_size=1)
        
        return loss

    def validation_step(self, val_batch, batch_idx):        
        pos, light, y, edge_index, batch = (
            val_batch.pos,
            val_batch.light,
            val_batch.label,
            val_batch.edge_index,
            val_batch.batch,
       )
        
        pd = self.model(light, pos, edge_index,val_batch.batch)
        loss = self.loss(pd, y)
        val_acc = acc(pd, y)
        self.log('val_loss', loss, batch_size=1)
        self.log('val_acc', val_acc, on_epoch=True, prog_bar=True, logger=True, batch_size=1)
        

    def predict_step(self, batch, batch_idx):
        pos, light, y, edge_index, batch, name = (
            batch.pos,
            batch.light,
            batch.label,
            batch.edge_index,
            batch.batch,
            batch.name
        )   
        
        pd = self.model(light, pos, edge_index,batch)
        return pd, y, name

# main

if __name__ == '__main__':
    train_dataset = AtomDataset_vor_aug(root='{}/train/'.format(DATA_PATH), data_type='train')
    valid_dataset = AtomDataset_vor_aug(root='{}/val/'.format(DATA_PATH), data_type='val')
    test_dataset = AtomDataset_vor_aug(root='{}/test/'.format(DATA_PATH), data_type='val')
    e2e_dataset = AtomDataset_vor_aug(root='{}/test/'.format(DATA_PATH), data_type='val')

    datamodule = LightningDataset(
        train_dataset,
        valid_dataset,
        test_dataset,
        e2e_dataset,
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
        min_epochs=20 ,
        logger = logger,
        callbacks = [earlystop_callback, checkpoint_callback],
    )

    trainer.fit(
        model, 
        datamodule,
    )
    
    # inference test
    predictions = trainer.predict(
        model,
        datamodule.test_dataset,
        ckpt_path = 'best',
    )
    save_results(trainer.log_dir, predictions, 'test')
    
    # inference e2e
    predictions = trainer.predict(
        model, 
        datamodule.pred_dataset,
        ckpt_path = 'best',
    )

    save_results(trainer.log_dir, predictions, 'e2e')
