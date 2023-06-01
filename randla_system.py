# torch
from typing import Any, Optional
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, STEP_OUTPUT
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
import torch.utils.data

# torch lightning
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers
from pytorch_lightning.strategies import DDPStrategy

# lib
import numpy as np
import pandas as pd
import os
import argparse
from datetime import datetime
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import time

# mine
from dataset.data import data_loaders
from models.model import RandLANet
from utils.tools import Config as cfg
from utils.metrics import accuracy, intersection_over_union
from utils.ply import read_ply, write_ply

def evaluate(model, points):
    model.eval()
    with torch.no_grad():
        scores = model(points)
    return scores

class RandLA_System(pl.LightningModule):

    def __init__(self, hparams):
        super(RandLA_System, self).__init__()
        self.hparams = hparams
        self.save_hyperparameters()
        try:
            with open(hparams.dataset / 'classes.json') as f:
                self.labels = json.load(f)
                self.num_classes = len(self.labels.keys())
        except FileNotFoundError:
            self.num_classes = int(input("Number of distinct classes in the dataset: "))
        self.weights = self.get_weights()
        self.loss = nn.CrossEntropyLoss(weight=self.weights)

    def get_weights(self):
        print('Computing weights...', end='\t')
        samples_per_class = np.array(cfg.class_weights)
        n_samples = torch.tensor(cfg.class_weights, dtype=torch.float)
        ratio_samples = n_samples / n_samples.sum()
        weights = 1 / (ratio_samples + 0.02)
        print('Done.')
        print('Weights:', weights)
        return weights
    
    def forward(self, input):
        return self.model(input)

    def decode_batch(self, batch):
        return batch

    def get_lr(self, optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    def setup(self, stage):
        if self.hparams.work_type == 'train':
            self.train_loader, self.val_loader = data_loaders(os.path.join(self.hparams.dataset, self.hparams.train_dir),
                self.hparams.dataset_sampling,
                batch_size=self.hparams.batch_size,
                num_workers=self.hparams.num_workers,
                pin_memory=True
            )
            d_in = next(iter(self.train_loader))[0].size(-1)
            self.model = RandLANet(
                d_in,
                self.num_classes,
                num_neighbors=self.hparams.neighbors,
                decimation=self.hparams.decimation,
            )
        elif self.hparams.work_type == 'test':
            print('Loading data...')
            self.test_loader, _ = data_loaders(
                os.path.join(self.hparams.dataset,self.hparams.test_dir),                
                self.hparams.dataset_sampling,
                batch_size=self.hparams.batch_size,
                num_workers=self.hparams.num_workers,
                pin_memory=True
            )
            d_in = 6
            num_classes = 14
            self.model = RandLANet(d_in, num_classes, 16, 4)
    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader
    
    def test_dataloader(self):
        return self.test_loader

    def configure_optimizers(self):
        # Load the Adam optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.adam_lr)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, self.hparams.scheduler_gamma)
        return [self.optimizer], [self.scheduler]

    def training_step(self, batch, batch_nb):
        points, labels = self.decode_batch(batch)
        scores = self(points)

        logp = torch.distributions.utils.probs_to_logits(scores, is_binary=False)

        loss = self.loss(logp, labels)
        accuracy = accuracy(scores, labels)
        iou = intersection_over_union(scores, labels)

        log= {'lr': self.get_lr(self.optimizer), 'train/loss': loss, 'train/accuracy': accuracy, 'train/iou': iou}
        
        return {'loss': loss, 'accuracy': accuracy,'iou': iou, 'log': log}

    def training_epoch_end(self, outputs):
        mean_loss = torch.stack([x['loss'] for x in outputs]).mean()
        mean_accuracy = torch.stack([x['accuracy'] for x in outputs]).mean()
        mean_iou = torch.stack([x['iou'] for x in outputs]).mean()
        self.log('mean_train_loss', mean_loss.clone().detach(), sync_dist=True)
        self.log('mean_train_accuracy', mean_accuracy.clone().detach(), sync_dist=True)
        self.log('mean_train_iou', mean_iou.clone().detach(), sync_dist=True)

    def validation_step(self, batch, batch_nb):
        self.model.eval()
        points, labels = self.decode_batch(batch)
        with torch.no_grad():
            scores = self(points)
        logp = torch.distributions.utils.probs_to_logits(scores, is_binary=False)

        loss = self.loss(logp, labels)
        accuracy = accuracy(scores, labels)
        iou = intersection_over_union(scores, labels)

        log = {'val/loss': loss, 'val/accuracy': accuracy, 'val/iou': iou}

        return {'loss': loss, 'accuracy': accuracy,'iou': iou, 'log': log}

    def validation_epoch_end(self, outputs):
        mean_loss = torch.stack([x['loss'] for x in outputs]).mean()
        mean_accuracy = torch.stack([x['accuracy'] for x in outputs]).mean()
        mean_iou = torch.stack([x['iou'] for x in outputs]).mean()
        self.log('mean_val_loss', mean_loss.clone().detach(), sync_dist=True)
        self.log('mean_val_accuracy', mean_accuracy.clone().detach(), sync_dist=True)
        self.log('mean_val_iou', mean_iou.clone().detach(), sync_dist=True)

    def test_step(self, batch, batch_nb):
        points, labels = self.decode_batch(batch)
        with torch.no_grad():
            scores = self(points)
        predictions = torch.max(scores, dim=-2).indices
        accuracy = (predictions == labels).float().mean()
        print('Accuracy:', accuracy.item())
        predictions = predictions.cpu().numpy()
        cloud = points.squeeze(0)[:,:3]
        write_ply('MiniDijon9.ply', [cloud, predictions], ['x', 'y', 'z', 'class'])


def train(args):
    if args.load == '':
        checkpoint = None
    else:
        checkpoint = ModelCheckpoint(dirpath=os.path.join(os.path.abspath(f'./ckpts/{args.name}')),
                                    filename='best',
                                    monitor='mean_train_loss',
                                    mode='min',
                                    save_top_k=5,
                                    )
    system = RandLA_System(hparams=args)
    trainer = pl.Trainer(max_epochs=args['epochs'],
                         callbacks=[checkpoint] if checkpoint is not None else None,
                         accelerator='gpu',devices=args.gpu,
                         strategy=DDPStrategy(find_unused_parameters=False),
                         check_val_every_n_epoch=5,
                         num_sanity_val_steps=1,
                         benchmark=True,
                         log_every_n_steps=1)
    trainer.fit(system)

def test(args):
    system = RandLA_System.load_from_checkpoint(args.load)
    trainer = pl.Trainer()
    trainer.test(system)

if __name__ == '__main__':

    """Parse program arguments"""
    parser = argparse.ArgumentParser(
        prog='RandLA-Net',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    base = parser.add_argument_group('Base options')
    expr = parser.add_argument_group('Experiment parameters')
    param = parser.add_argument_group('Hyperparameters')
    dirs = parser.add_argument_group('Storage directories')
    misc = parser.add_argument_group('Miscellaneous')

    base.add_argument('--dataset', type=Path, help='location of the dataset',
                        default='datasets/s3dis/subsampled')
    
    base.add_argument('--work_type', type=str, help='train, val, test', default='train')

    expr.add_argument('--epochs', type=int, help='number of epochs',
                        default=50)
    expr.add_argument('--load', type=str, help='model to load',
                        default='')

    param.add_argument('--adam_lr', type=float, help='learning rate of the optimizer',
                        default=1e-2)
    param.add_argument('--batch_size', type=int, help='batch size',
                        default=10)
    param.add_argument('--decimation', type=int, help='ratio the point cloud is divided by at each layer',
                        default=4)
    param.add_argument('--dataset_sampling', type=str, help='how dataset is sampled',
                        default='active_learning', choices=['active_learning', 'naive'])
    param.add_argument('--neighbors', type=int, help='number of neighbors considered by k-NN',
                        default=16)
    param.add_argument('--scheduler_gamma', type=float, help='gamma of the learning rate scheduler',
                        default=0.95)

    dirs.add_argument('--test_dir', type=str, help='location of the test set in the dataset dir',
                        default='test')
    dirs.add_argument('--train_dir', type=str, help='location of the training set in the dataset dir',
                        default='train')
    dirs.add_argument('--val_dir', type=str, help='location of the validation set in the dataset dir',
                        default='val')
    dirs.add_argument('--logs_dir', type=Path, help='path to tensorboard logs',
                        default='runs')

    misc.add_argument('--gpu', type=int, help='which GPU to use (-1 for CPU)',
                        default=2)
    misc.add_argument('--name', type=str, help='name of the experiment',
                        default=None)
    misc.add_argument('--num_workers', type=int, help='number of threads for loading data',
                        default=4)
    misc.add_argument('--save_freq', type=int, help='frequency of saving checkpoints',
                        default=10)

    args = parser.parse_args()


    if args.name is None:
        if args.load:
            args.name = args.load
        else:
            args.name = datetime.now().strftime('%Y-%m-%d_%H:%M')

    if args.work_type == 'train':
        train(args)
    elif args.work_type == 'test':
        test(args)
    

