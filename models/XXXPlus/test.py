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
model = XXXPlusModel(user_params, item_params, 48, 128, [128, 64, 32],
                [4, 4], loss_fn, activation)
opt = Adam(model.parameters(), lr=0.0005)


===
model = FedXXXLaunch(user_params, item_params, 48, 128, [128,64,32], -1,
                     [2,2], activation, train_data, test_data, loss_fn, 5,
                     [160], "my_layer", 1)  5% 0.41mae
5% Epoch:150 mae:0.4128972291946411,mse:2.055781841278076,rmse:1.4337997436523438
mae:0.39854055643081665,mse:1.9525188207626343,rmse:1.3973256349563599
20%  Epoch:260 mae:0.35921603441238403,mse:1.7138055562973022,rmse:1.3091239929199219
===



4.2日
20


196313 20 Epoch:200 mae:0.34299609065055847,mse:1.6866803169250488,rmse:1.2987226247787476
model = FedXXXLaunch(user_params, item_params, 48, 128, [128,64,32], -1,
                     [3,3], activation, train_data, test_data, loss_fn, 5,
                     [160,32], "my_layer", 1)
5% mae 0.39左右



============================
model = FedXXXLaunch(user_params, item_params, 48, 128, [128,64,32,16], -1,
                     [3,3,2], activation, train_data, test_data, loss_fn, 5,
                     [144,64], "my_layer", 1)
FedPer

5% mae:0.394509494304657,mse:1.979100227355957,rmse:1.4068049192428589
10% [density:0.1,type:rt] Epoch:290 mae:0.362935870885849,mse:1.7823574542999268,rmse:1.3350496292114258
15% [density:0.15,type:rt] Epoch:200 mae:0.35469549894332886,mse:1.7218421697616577,rmse:1.3121898174285889
20% Epoch:290 mae:0.3350681960582733,mse:1.5960551500320435,rmse:1.2633507251739502

5% [density:0.05,type:tp] Epoch:270 mae:18.568113327026367,mse:3464.732421875,rmse:58.861976623535156
10% Epoch:120 mae:16.582380294799805,mse:2846.251220703125,rmse:53.35026931762695
15%  mae:16.856727600097656,mse:3118.36962890625,rmse:55.84236526489258
[density:0.2,type:tp] Epoch:160 mae:16.046567916870117,mse:2939.0791015625,rmse:54.21327590942383

# 用上述参数进行非个性化的联邦学习训练


5% Epoch:180 mae:0.5083925724029541,mse:2.5896036624908447,rmse:1.609224557876587
10% [density:0.1,type:rt] Epoch:220 mae:0.4981437921524048,mse:2.5294189453125,rmse:1.5904147624969482
15% Epoch:240 mae:0.5026066899299622,mse:2.497178792953491,rmse:1.5802464485168457
[density:0.2,type:rt] Epoch:140 mae:0.47787535190582275,mse:2.226895332336426,rmse:1.4922785758972168

%5 [density:0.05,type:tp] Epoch:90 mae:21.782121658325195,mse:4895.1015625,rmse:69.96500396728516
10% mae:20.48955535888672,mse:3420.38037109375,rmse:58.48401641845703
[density:0.15,type:tp] Epoch:150 mae:19.120758056640625,mse:3165.0810546875,rmse:56.25905227661133
[density:0.2,type:tp] Epoch:110 mae:18.962905883789062,mse:3085.902587890625,rmse:55.550899505615234
=========================

2022年04月09日
因为对于tp无法拟合 增加参数量
model = FedXXXLaunch(user_params, item_params, 48, 128, [256, 128, 64, 32, 16], -1,
                     [3, 4, 3, 2], activation, train_data, test_data, loss_fn, 5,
                     [144, 64], "my_layer", 1)

FedRep
header_epoch 10
model = FedXXXLaunch(user_params, item_params, 48, 128, [128, 64, 32, 16],
                        -1, [3, 3, 2], activation, train_data, test_data,
                        loss_fn, 1, [144, 64], True, 10, "my_layer", 1)



"""

epochs = 3000
density = 0.2
type_ = "rt"

is_fed = False
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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
loss_fn = nn.L1Loss()

activation = nn.GELU

user_params = {
    "type_": "cat",  # embedding层整合方式 stack or cat
    "embedding_nums": u_info.embedding_nums,  # 每个要embedding的特征的总个数
    "embedding_dims": [4, 4, 4],
}

item_params = {
    "type_": "cat",  # embedding层整合方式 stack or cat
    "embedding_nums": i_info.embedding_nums,  # 每个要embedding的特征的总个数
    "embedding_dims": [4, 4, 4],
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
        "blocks_size": [128, 64, 32, 16],
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

    model.fit(epochs, 0.0005, 10, 1, f"density:{density},type:{type_}")

else:

    train_data = data_preprocess(train, u_info, i_info)
    test_data = data_preprocess(test, u_info, i_info)
    train_dataset = ToTorchDataset(train_data)
    test_dataset = ToTorchDataset(test_data)
    train_dataloader = DataLoader(train_dataset, batch_size=512)
    test_dataloader = DataLoader(test_dataset, batch_size=2048)


    model = XXXPlusModel(user_params, item_params, 24, 128, [64, 32, 16],
                         [3,3,3], loss_fn, activation, [144,64,32])
    print(f"模型参数:", count_parameters(model))
    
    opt = Adam(model.parameters(), lr=0.0005)
    # opt = SGD(model.parameters(), lr=0.01)

    model.fit(train_dataloader,
              epochs,
              opt,
              eval_loader=test_dataloader,
              save_filename=f"{density}_{type_}")
