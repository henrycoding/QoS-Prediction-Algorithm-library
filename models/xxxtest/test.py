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



"""



config = {

    "CUDA_VISIBLE_DEVICES":"0",
    "embedding_dims":[64,32,32],
    "density":0.2,
    "type_":"tp",
    "epoch":400,
    "is_fed":False,
    "train_batch_size":256,
    "lr":0.001,
    "in_size":None,
    "out_size":None,
    "blocks":[256,128,64],
    "deepths":[1,1,1],
    "linear_layer":[320,64],
    "weight_decay":0,
    "loss_fn":nn.L1Loss()
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

activation = nn.ReLU
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

if is_fed:

    fed_data_preprocess = partial(data_preprocess, is_dtriad=True)

    train_data = fed_data_preprocess(train, u_info, i_info)
    test_data = fed_data_preprocess(test, u_info, i_info)

    params = {
        "user_embedding_params": user_params,
        "item_embedding_params": item_params,
        "in_size": 4*6,
        "output_size": 128,
        "blocks_size": [64, 32, 16],
        "batch_size": -1,
        "deepths": [3,3,3],
        "activation": activation,
        "d_triad": train_data,
        "test_d_triad": test_data,
        "loss_fn": loss_fn,
        "local_epoch": 5,
        "linear_layers": [144, 32],
        "is_personalized": False,
        "header_epoch": None,
        "personal_layer": "my_layer",
        "output_dim": 1,
        "optimizer": "adam",
        "use_gpu": True
    }

    model = FedXXXLaunch(**params)
    print(f"模型参数:", count_parameters(model))

    model.fit(epochs, config["lr"], 10, 1, f"density:{density},type:{type_}")

else:

    train_data = data_preprocess(train, u_info, i_info)
    test_data = data_preprocess(test, u_info, i_info)
    train_dataset = ToTorchDataset(train_data)
    test_dataset = ToTorchDataset(test_data)
    train_dataloader = DataLoader(train_dataset, batch_size=config["train_batch_size"])
    test_dataloader = DataLoader(test_dataset, batch_size=2048)


    model = XXXPlusModel(user_params, item_params, config["in_size"], config["out_size"], config["blocks"],
                         config["deepths"], loss_fn, activation, config["linear_layer"])
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
