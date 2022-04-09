import os
import shutil
import sys
import time

import torch
from data import MatrixDataset, ToTorchDataset
from models.NeuMF.model import NeuMF, NeuMFModel
from root import absolute
from torch import nn, optim
from torch.nn.modules import loss
from torch.optim import Adam
from torch.utils.data import DataLoader
from utils.evaluation import mae, mse, rmse
from root import ROOT
from models.NeuMF.model import NeuMF, NeuMFModel

# 冻结随机数
from utils.model_util import freeze_random

from torch.utils.tensorboard import SummaryWriter
from tensorboard import program

from utils.model_util import TensorBoardTool

"""
RESULT NeuMF:
Density:0.05, type:rt, mae:0.6073, mse:3.6153, rmse:1.9014
Density:0.10, type:rt, mae:0.5820, mse:3.5959, rmse:1.8963
Density:0.15, type:rt, mae:0.5713, mse:3.5806, rmse:1.8922
Density:0.20, type:rt, mae:0.5837, mse:3.6152, rmse:1.9014


          0.05      0.10      0.15      0.20
MAE   0.582943  0.595642  0.580388  0.571211
MSE   3.598602  3.594801  3.588758  3.601026
RMSE  1.896998  1.895996  1.894402  1.897637


          0.05      0.10      0.15      0.20
MAE   0.581534  0.595348  0.580146  0.568814
MSE   3.600061  3.603946  3.591941  3.587708
RMSE  1.897383  1.898406  1.895242  1.894125
"""

freeze_random()  # 冻结随机数 保证结果一致

tb_tool = TensorBoardTool(os.getcwd())
writer = tb_tool.run()

for density in [0.05, 0.1, 0.15, 0.2]:
    for type_ in ['rt', 'tp']:
        rt_data = MatrixDataset(type_)
        train_data, test_data = rt_data.split_train_test(density)

        train_dataset = ToTorchDataset(train_data)
        test_dataset = ToTorchDataset(test_data)
        train_dataloader = DataLoader(train_dataset, batch_size=64)
        test_dataloader = DataLoader(test_dataset, batch_size=64)

        lr = 0.001
        epochs = 2
        dim = 8

        loss_fn = nn.L1Loss()
        NeuMF = NeuMFModel(loss_fn, rt_data.row_n, rt_data.col_n, dim, density, writer=writer)
        opt = Adam(NeuMF.parameters(), lr=lr, weight_decay=1e-4)

        NeuMF.fit(train_dataloader, epochs, opt, eval_loader=test_dataloader)

        y, y_pred = NeuMF.predict(test_dataloader)
        mae_ = mae(y, y_pred)
        mse_ = mse(y, y_pred)
        rmse_ = rmse(y, y_pred)

        NeuMF.logger.info(
            f"Density:{density:.2f}, type:{type_}, mae:{mae_:.4f}, mse:{mse_:.4f}, rmse:{rmse_:.4f}")
