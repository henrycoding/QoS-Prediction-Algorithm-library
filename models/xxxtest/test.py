from collections import namedtuple
from functools import partial

import numpy as np
import torch
from data import InfoDataset, MatrixDataset, ToTorchDataset
from models.XXXPlus.model import XXXPlus
from models.XXXPlus.resnet_utils import ResNetBasicBlock
from torch import nn, optim
from torch.nn.modules import loss
from torch.optim import SGD, Adam, optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.decorator import timeit
from utils.evaluation import mae, mse, rmse
from utils.model_util import count_parameters, freeze_random

from .model import FedXXXLaunch, XXXPlusModel
"""
Fed:

2,2,2

[density:0.05,type:rt] Epoch:60 mae:0.39640194177627563,mse:1.8226033449172974,rmse:1.3500382900238037
[density:0.1,type:rt] Epoch:60 mae:0.36088812351226807,mse:1.638769268989563,rmse:1.280144214630127
[density:0.2,type:rt] Epoch:60 mae:0.3386975824832916,mse:1.5334124565124512,rmse:1.2383103370666504

[density:0.05,type:tp] Epoch:100 mae:16.623676300048828,mse:3133.034912109375,rmse:55.97351837158203
[density:0.1,type:tp] Epoch:100 mae:14.519242286682129,mse:2430.328369140625,rmse:49.29835891723633
[density:0.15,type:tp] Epoch:100 mae:13.690592765808105,mse:2160.440673828125,rmse:46.48054122924805
[density:0.2,type:tp] Epoch:100 mae:14.106036186218262,mse:2230.414794921875,rmse:47.22726821899414

4 4 4 
[density:0.05,type:rt] Epoch:60 mae:0.39592623710632324,mse:1.803076982498169,rmse:1.3427870273590088
[density:0.1,type:rt] Epoch:60 mae:0.3592352271080017,mse:1.6179782152175903,rmse:1.2719976902008057
[density:0.15,type:rt] Epoch:80 mae:0.3353453576564789,mse:1.5067840814590454,rmse:1.2275112867355347
[density:0.2,type:rt] Epoch:80 mae:0.3254038989543915,mse:1.4776121377944946,rmse:1.2155706882476807

8 8 8
[density:0.05,type:rt] Epoch:35 mae:0.3985772728919983,mse:1.797226905822754,rmse:1.340606927871704
[density:0.1,type:rt] Epoch:55 mae:0.35645025968551636,mse:1.5757331848144531,rmse:1.2552821636199951
[density:0.15,type:rt] Epoch:75 mae:0.3352337181568146,mse:1.4594119787216187,rmse:1.2080612182617188
[density:0.2,type:rt] Epoch:85 mae:0.31767740845680237,mse:1.4006195068359375,rmse:1.183477759361267


"""

config = {
    "CUDA_VISIBLE_DEVICES": "0",
    "embedding_dims": [2,2,2],
    "density": 0.05,
    "type_": "tp",
    "epoch": 4000,
    "is_fed": True,
    "train_batch_size": 256,
    "lr": 0.001,
    "in_size": 2 * 6,
    "out_size": None,
    "blocks": [256,128,64],
    "deepths": [1,1,1],
    "linear_layer": [320,64],
    "weight_decay": 0,
    "loss_fn": nn.L1Loss(),
    "is_personalized": True,
    "activation": nn.ReLU,
    "select":1,
    "local_epoch": 5,
    "fed_bs": -1
    # "备注":"embedding初始化参数0,001"
}


epochs = config["epoch"]
density = config["density"]
type_ = config["type_"]

is_fed = config["is_fed"]
import os

os.environ["CUDA_VISIBLE_DEVICES"] = config["CUDA_VISIBLE_DEVICES"]


def data_preprocess(triad,
                    u_info_obj: InfoDataset,
                    i_info_obj: InfoDataset,
                    is_dtriad=False):
    """解决uid和embedding时的id不一致的问题
    生成d_triad [[triad],[p_triad]]
    """
    r = []
    for row in tqdm(triad, desc="Gen d_triad", ncols=80):
        uid, iid, rate = int(row[0]), int(row[1]), float(row[2])
        u = u_info_obj.query(uid)
        i = i_info_obj.query(iid)
        r.append([[uid, iid, rate], [u, i, rate]]) if is_dtriad else r.append(
            [u, i, rate])
    return r


u_enable_columns = ["[User ID]", "[Country]", "[AS]"]
i_enable_columns = ["[Service ID]", "[Country]", "[AS]"]

fed_data_preprocess = partial(data_preprocess, is_dtriad=True)

md = MatrixDataset(type_)
u_info = InfoDataset("user", u_enable_columns)
i_info = InfoDataset("service", i_enable_columns)
train, test = md.split_train_test(density)

# loss_fn = nn.SmoothL1Loss()
# loss_fn = nn.L1Loss()
loss_fn = config["loss_fn"]

activation = config["activation"]
# activation = nn.ReLU

user_params = {
    "type_": "cat",  # embedding层整合方式 stack or cat
    "embedding_nums": u_info.embedding_nums,  # 每个要embedding的特征的总个数
    "embedding_dims": config["embedding_dims"],
}

item_params = {
    "type_": "cat",  # embedding层整合方式 stack or cat
    "embedding_nums": i_info.embedding_nums,  # 每个要embedding的特征的总个数
    "embedding_dims": config["embedding_dims"],
}

if config["is_fed"]:

    fed_data_preprocess = partial(data_preprocess, is_dtriad=True)

    train_data = fed_data_preprocess(train, u_info, i_info)
    test_data = fed_data_preprocess(test, u_info, i_info)

    params = {
        "user_embedding_params": user_params,
        "item_embedding_params": item_params,
        "in_size": config["in_size"],
        "output_size": config["out_size"],
        "blocks_size": config["blocks"],
        "batch_size": config["fed_bs"],
        "deepths": config["deepths"],
        "activation": activation,
        "d_triad": train_data,
        "test_d_triad": test_data,
        "loss_fn": config["loss_fn"],
        "local_epoch": config["local_epoch"],
        "linear_layers": config["linear_layer"],
        "is_personalized": config["is_personalized"],
        "header_epoch": None,
        "personal_layer": "my_layer",
        "output_dim": 1,
        "optimizer": "adam",
        "use_gpu": True
    }

    model = FedXXXLaunch(**params)
    print(f"模型参数:", count_parameters(model))
    print(model)
    print(config)
    model.fit(epochs, config["lr"], 5, config["select"], f"density:{density},type:{type_}")

else:

    train_data = data_preprocess(train, u_info, i_info)
    test_data = data_preprocess(test, u_info, i_info)
    train_dataset = ToTorchDataset(train_data)
    test_dataset = ToTorchDataset(test_data)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=config["train_batch_size"])
    test_dataloader = DataLoader(test_dataset, batch_size=2048)

    model = XXXPlusModel(user_params, item_params, config["in_size"],
                         config["out_size"], config["blocks"],
                         config["deepths"], loss_fn, activation,
                         config["linear_layer"])
    print(f"模型参数:", count_parameters(model))
    print(model)
    print(config)
    
    opt = Adam(model.parameters(), lr=config["lr"],weight_decay=config["weight_decay"])
    # opt = SGD(model.parameters(), lr=0.01)

    model.fit(train_dataloader,
              epochs,
              opt,
              eval_loader=test_dataloader,
              save_filename=f"{density}_{type_}")
