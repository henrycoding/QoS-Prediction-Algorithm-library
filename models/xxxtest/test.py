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
[density:0.15,type:tp] Epoch:80 mae:13.745682716369629,mse:2111.67578125,rmse:45.952972412109375
[density:0.2,type:tp] Epoch:80 mae:13.631184577941895,mse:2176.6337890625,rmse:46.6544075012207

4 4 4 
[density:0.05,type:rt] Epoch:60 mae:0.39592623710632324,mse:1.803076982498169,rmse:1.3427870273590088
[density:0.1,type:rt] Epoch:60 mae:0.3592352271080017,mse:1.6179782152175903,rmse:1.2719976902008057
[density:0.15,type:rt] Epoch:80 mae:0.3353453576564789,mse:1.5067840814590454,rmse:1.2275112867355347
[density:0.2,type:rt] Epoch:80 mae:0.3254038989543915,mse:1.4776121377944946,rmse:1.2155706882476807

[density:0.05,type:tp] Epoch:80 mae:15.746493339538574,mse:2869.870849609375,rmse:53.57117462158203
[density:0.1,type:tp] Epoch:80 mae:13.736608505249023,mse:2279.198486328125,rmse:47.74095153808594
[density:0.15,type:tp] Epoch:80 mae:12.842589378356934,mse:1937.2783203125,rmse:44.014522552490234
[density:0.2,type:tp] Epoch:80 mae:12.18262767791748,mse:1775.5418701171875,rmse:42.13718032836914

8 8 8
[density:0.05,type:rt] Epoch:35 mae:0.3985772728919983,mse:1.797226905822754,rmse:1.340606927871704
[density:0.1,type:rt] Epoch:55 mae:0.35645025968551636,mse:1.5757331848144531,rmse:1.2552821636199951
[density:0.15,type:rt] Epoch:75 mae:0.3352337181568146,mse:1.4594119787216187,rmse:1.2080612182617188
[density:0.2,type:rt] Epoch:85 mae:0.31767740845680237,mse:1.4006195068359375,rmse:1.183477759361267

[density:0.05,type:tp] Epoch:90 mae:15.565387725830078,mse:2762.900146484375,rmse:52.563297271728516
[density:0.1,type:tp] Epoch:90 mae:12.96229362487793,mse:2074.105712890625,rmse:45.54235076904297
[density:0.15,type:tp] Epoch:90 mae:12.154128074645996,mse:1834.3294677734375,rmse:42.829071044921875
[density:0.2,type:tp] Epoch:90 mae:11.706123352050781,mse:1700.9140625,rmse:41.24213790893555


16 16 16
[density:0.05,type:rt] Epoch:65 mae:0.4096967577934265,mse:1.8146902322769165,rmse:1.3471044301986694
[density:0.1,type:rt] Epoch:75 mae:0.35850274562835693,mse:1.5438477993011475,rmse:1.2425167560577393
[density:0.15,type:rt] Epoch:75 mae:0.33630967140197754,mse:1.4385364055633545,rmse:1.1993900537490845
[density:0.2,type:rt] Epoch:65 mae:0.31893444061279297,mse:1.3725398778915405,rmse:1.171554446220398

[density:0.05,type:tp] Epoch:80 mae:15.342988967895508,mse:2699.822509765625,rmse:51.959815979003906
[density:0.1,type:tp] Epoch:80 mae:12.7587251663208,mse:2020.516845703125,rmse:44.95016098022461
[density:0.15,type:tp] Epoch:80 mae:11.635480880737305,mse:1686.2298583984375,rmse:41.06372833251953
[density:0.2,type:tp] Epoch:80 mae:11.280876159667969,mse:1615.932861328125,rmse:40.198665618896484

32 32 32

32 32 32
[density:0.05,type:tp] Epoch:160 mae:14.670263290405273,mse:2443.43310546875,rmse:49.431095123291016
[density:0.1,type:tp] Epoch:160 mae:13.012394905090332,mse:2018.0687255859375,rmse:44.92292022705078
[density:0.15,type:tp] Epoch:160 mae:11.701943397521973,mse:1701.3505859375,rmse:41.247432708740234
[density:0.2,type:tp] Epoch:160 mae:11.205389976501465,mse:1614.933349609375,rmse:40.18623352050781

[density:0.05,type:rt] Epoch:160 mae:0.40601715445518494,mse:1.773358702659607,rmse:1.3316751718521118
[density:0.1,type:rt] Epoch:160 mae:0.35660964250564575,mse:1.5247416496276855,rmse:1.2348042726516724
[density:0.15,type:rt] Epoch:160 mae:0.33663177490234375,mse:1.421412706375122,rmse:1.1922301054000854


Non-Fed
[0.05_rt] Epoch:200 mae:0.3763183653354645,mse:1.8360363245010376,rmse:1.3550041913986206
[0.1_rt] Epoch:200 mae:0.32560423016548157,mse:1.5566685199737549,rmse:1.247665286064148
[0.15_rt] Epoch:200 mae:0.30293840169906616,mse:1.4107856750488281,rmse:1.1877650022506714
[0.2_rt] Epoch:200 mae:0.28960633277893066,mse:1.3320720195770264,rmse:1.1541543006896973

[0.05_tp] Epoch:200 mae:13.811687469482422,mse:2294.42333984375,rmse:47.90013885498047
[0.1_tp] Epoch:200 mae:11.252021789550781,mse:1666.812255859375,rmse:40.82661056518555
[0.15_tp] Epoch:200 mae:10.421308517456055,mse:1445.4444580078125,rmse:38.01900100708008
[0.2_tp] Epoch:200 mae:10.136045455932617,mse:1378.9722900390625,rmse:37.134517669677734
"""

config = {
    "CUDA_VISIBLE_DEVICES": "0",
    "embedding_dims": [64,64,64],
    "density": 0.1,
    "type_": "rt",
    "epoch": 4000,
    "is_fed": False,
    "train_batch_size": 256,
    "lr": 0.001,
    "in_size": 16 * 6,
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
