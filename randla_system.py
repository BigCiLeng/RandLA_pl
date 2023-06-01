# torch
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

# lib
import numpy as np
import pandas as pd
import os

# mine
from dataset.semkitti_trainset import SemanticKITTI
from utils.config import ConfigSemanticKITTI as cfg
from utils.metric import compute_acc, IoUCalculator
from models.RandLANet import Network
from models.loss_func import compute_loss

torch.backends.cudnn.enabled = False

def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

class RandLA_System(pl.LightningModule):

    def __init__(self, hparams):
        super(RandLA_System, self).__init__()
        self.hparams = hparams
        self.save_hyperparameters()

        self.net = Network(cfg)

    def forward(self, input):
        result = self.mlp(input)
        return result

    def decode_batch(self, batch):
        data = batch['data']
        label = batch['label']
        return data, label

    def test_decode_batch(self, batch):
        data = batch['data']
        return data

    def get_lr(self, optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    def setup(self, stage):
        self.train_dataset = SemanticKITTI('training')
        self.val_dataset = SemanticKITTI('validation')
        class_weights = torch.from_numpy(self.train_dataset.get_class_weight()).float().cuda()
        self.criterion = nn.CrossEntropyLoss(weight=class_weights, reduction='none')

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.hparams.batch_size,
                          shuffle=True,
                          num_workers=self.hparams.num_workers,
                          worker_init_fn=my_worker_init_fn,
                          collate_fn=self.train_dataset.collate_fn,
                          pin_memory=True
                        )

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.hparams.val_batch_size,
                          shuffle=True,
                          num_workers=self.hparams.num_workers,
                          worker_init_fn=my_worker_init_fn,
                          collate_fn=self.val_dataset.collate_fn,
                          pin_memory=True
                        )

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          shuffle=False,
                          num_workers=4,
                          batch_size=10,
                          pin_memory=True)

    def configure_optimizers(self):
        # Load the Adam optimizer
        self.optimizer = optim.Adam(self.net.parameters(), lr=cfg.learning_rate)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, 0.95)
        return [self.optimizer], [self.scheduler]

    def training_step(self, batch, batch_nb):
        log = {'lr': self.get_lr(self.optimizer)}
        data, label = self.decode_batch(batch)
        results = self(data).squeeze(1)
        log['train/loss'] = loss = self.loss(results, label)
        return {'loss': loss, 'log': log}

    def training_epoch_end(self, outputs):
        mean_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('mean_train_loss', mean_loss.clone().detach(), sync_dist=True)

    def validation_step(self, batch, batch_nb):
        data, label = self.decode_batch(batch)
        results = self(data).squeeze(1)
        log = {'val_loss': self.loss(results, label)}
        return log

    def validation_epoch_end(self, outputs):
        all_loss = outputs[0]['val_loss']
        mean_loss = all_loss.mean()
        return {'progress_bar': {'val_loss': mean_loss}, 'log': {'val/loss': mean_loss}}

    def test_step(self, batch, batch_idx):
        data, rank = self.decode_batch(batch)
        results = self(data).squeeze(0)
        return results

    def test_epoch_end(self, outputs):
        temp = torch.cat([i for i in outputs])
        result = denormalize(
            temp.numpy(), self.h['rank_1'], self.h['rank_2'])
        pd.DataFrame(result).to_csv(
            "data_output/output_0.csv", index=False, header=0)
        # with open(os.path.abspath(f'./logs/weight_opt/result'))



if __name__ == "__main__":
    print('ok')
