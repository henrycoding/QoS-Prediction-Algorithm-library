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




model = FedXXXLaunch(user_params, item_params, 48, 128, [128,64,32,16], -1,
                     [3,3,2], activation, train_data, test_data, loss_fn, 5,
                     [144,64], "my_layer", 1)
5% mae:0.394509494304657,mse:1.979100227355957,rmse:1.4068049192428589
20% Epoch:290 mae:0.3350681960582733,mse:1.5960551500320435,rmse:1.2633507251739502


"""

epochs = 3000
density = 0.2
type_ = "rt"


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
    "embedding_dims": [8, 8, 8],
}

item_params = {
    "type_": "cat",  # embedding层整合方式 stack or cat
    "embedding_nums": i_info.embedding_nums,  # 每个要embedding的特征的总个数
    "embedding_dims": [8, 8, 8],
}

fed_data_preprocess = partial(data_preprocess, is_dtriad=True)

train_data = fed_data_preprocess(train, u_info, i_info)
test_data = fed_data_preprocess(test, u_info, i_info)

# train_data = data_preprocess(train, u_info, i_info)
# test_data = data_preprocess(test, u_info, i_info)
# train_dataset = ToTorchDataset(train_data)
# test_dataset = ToTorchDataset(test_data)
# train_dataloader = DataLoader(train_dataset, batch_size=128)
# test_dataloader = DataLoader(test_dataset, batch_size=2048)

model = FedXXXLaunch(user_params, item_params, 48, 128, [128,64,32,16], -1,
                     [3,3,2], activation, train_data, test_data, loss_fn, 5,
                     [144,64], "my_layer", 1)
print(f"模型参数:", count_parameters(model))

model.fit(epochs, 0.0005, 1, f"density:{density},type:{type_}")

# model = XXXPlusModel(user_params, item_params, 48, 128, [128, 64, 32], [4, 4],
#                      loss_fn, activation)
# opt = Adam(model.parameters(), lr=0.0005)
# opt = SGD(model.parameters(), lr=0.01)

# model.fit(train_dataloader,
#           epochs,
#           opt,
#           eval_loader=test_dataloader,
#           save_filename=f"{density}_{type_}")
