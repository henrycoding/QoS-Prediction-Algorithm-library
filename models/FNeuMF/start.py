import time

import torch
from data import MatrixDataset, ToTorchDataset
from models.FNeuMF.config import get_cfg_defaults

from root import absolute
from torch import nn, optim
from torch.nn.modules import loss
from torch.optim import Adam
from torch.utils.data import DataLoader
from utils.evaluation import mae, mse, rmse
from utils.model_util import freeze_random
from yacs.config import CfgNode

from model import FedNeuMFModel

"""
RESULT FedMLP:

"""

freeze_random()  # 冻结随机数 保证结果一致
cfg = get_cfg_defaults()
density_list = cfg.TRAIN.DENSITY_LIST
type_ = cfg.TRAIN.DATA_TYPE
rt_data = MatrixDataset(type_)
batch_size = cfg.TRAIN.BATCH_SIZE
lr = cfg.TRAIN.OPTIMIZER.LR
epochs = cfg.TRAIN.NUM_EPOCHS
dim = cfg.TRAIN.LATENT_DIM
fraction = cfg.TRAIN.FRACTION
layers = cfg.TRAIN.LAYERS
date = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())

for density in density_list:
    train_data, test_data = rt_data.split_train_test(density)

    train_dataset = ToTorchDataset(train_data)
    test_dataset = ToTorchDataset(test_data)

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)

    # loss_fn = nn.SmoothL1Loss()
    loss_fn = nn.L1Loss()

    neumf = FedNeuMFModel(
        train_data,
        loss_fn,
        rt_data.row_n,
        rt_data.col_n,
        dim=dim,
        layers=layers,
    )

    neumf.fit(epochs, lr, train_dataloader, date, density, fraction)
    y, y_pred = neumf.predict(
        test_dataloader, False,
    )
    mae_ = mae(y, y_pred)
    mse_ = mse(y, y_pred)
    rmse_ = rmse(y, y_pred)

    print(f"Density:{density},type:{type_},mae:{mae_},mse:{mse_},rmse:{rmse_}")
