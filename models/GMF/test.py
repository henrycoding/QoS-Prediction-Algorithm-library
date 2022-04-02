import os

import torch
from data import MatrixDataset, ToTorchDataset
from root import absolute
from torch import nn, optim
from torch.nn.modules import loss
from torch.optim import Adam
from torch.utils.data import DataLoader
from utils.evaluation import mae, mse, rmse
from utils.model_util import freeze_random

from model import GMFModel

from utils.model_util import TensorBoardTool

from models.GMF.config import get_cfg_defaults

"""
RESULT GMF:
"""

freeze_random()  # 冻结随机数 保证结果一致

# TensorBoard
tb_tool = TensorBoardTool(os.getcwd())
writer = tb_tool.run()

for density in [0.05, 0.1, 0.15, 0.2]:
    target_type = 'rt'
    rt_data, train_dataloader, test_dataloader = data_loading(density, target_type)

    cfg = get_cfg_defaults()
    opts = ["TRAIN.DENSITY", density]
    cfg.merge_from_list(opts)
    cfg.freeze()
    print(cfg)

    # gmf_config = {
    #     'name': 'GMF',
    #     'target_type': target_type,
    #     'density': density,
    #     'num_users': rt_data.row_n,
    #     'num_items': rt_data.col_n,
    #     'latent_dim': 8,
    #     'num_epoch': 100,
    #     'loss_fn': nn.L1Loss(),
    #     'optimizer': 'adam',
    #     'adam_lr': 1e-3,
    #     'l2_regularization': 0,
    #     'use_gpu': True,
    #     'device_id': 0,
    #     'writer': writer,
    # }



    # model = GMFModel(gmf_config)
    # model.fit(train_dataloader, eval_loader=test_dataloader)
    # y, y_pred = model.predict(test_dataloader)
    #
    # mae_ = mae(y, y_pred)
    # mse_ = mse(y, y_pred)
    # rmse_ = rmse(y, y_pred)
    #
    # model.logger.info(
    #     f"Density:{density}, type:{target_type}, mae:{mae_:.4f}, mse:{mse_:.4f}, rmse:{rmse_:.4f}")
