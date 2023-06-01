import numpy as np
import os
import shutil
import pytorch_lightning as pl
from evaluate_normalxy import MLP_EVALUATE_SYSTEM
from evaluate_normalxy import ori_rank
from evaluate_normalxy import ori_data_std
from pytorch_lightning.strategies import DDPStrategy
import json
if __name__ == '__main__':
     rank_1, rank_2 = ori_rank()
     data_1, data_2 = ori_data_std()
     with open('config.json', 'r') as f:  # 读配置文件，里面有最大进程数等
          h = json.load(f)

     h["rank_1"] = rank_1
     h["rank_2"] = rank_2
     h["data_1"] = data_1
     h["data_2"] = data_2

     if(os.path.exists(os.path.abspath(r'./ckpts/evaluate/'))):
          shutil.rmtree(os.path.abspath(r'./ckpts/evaluate/'))
     else:
          os.mkdir(os.path.abspath(r'./ckpts/evaluate/'))
     if(os.path.exists(os.path.abspath(r'./logs/evaluate'))):
          shutil.rmtree(os.path.abspath(r'./logs/evaluate'))
     else:
          os.mkdir(os.path.abspath(r'./logs/evaluate/'))

     system = MLP_EVALUATE_SYSTEM(hparams=h)
     # checkpoint = ModelCheckpoint(dirpath=os.path.join(os.path.abspath(f'./ckpts/{h["exp_name"]}')),
     #                             filename='best',
     #                             monitor='mean_train_loss',
     #                             mode='min',
     #                             save_top_k=5,
     #                             )

     # logger = loggers.CSVLogger(
     #     save_dir = 'logs',
     #     name = hparams['exp_name'],
     #     flush_logs_every_n_steps=10
     # )
     trainer = pl.Trainer(max_epochs=h['num_epochs'],
                         # callbacks=[checkpoint],
                         # logger=logger,
                         accelerator='gpu',devices=h['num_gpus'],
                         strategy=DDPStrategy(find_unused_parameters=False),
                         check_val_every_n_epoch=5,
                         num_sanity_val_steps=1,
                         benchmark=True,
                         profiler='sample' if h['num_gpus']==1 else None,
                         log_every_n_steps=1)
     trainer.fit(system)
