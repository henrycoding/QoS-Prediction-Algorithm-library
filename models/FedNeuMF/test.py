import torch
from data import MatrixDataset, ToTorchDataset
from root import absolute
from torch import nn, optim
from torch.nn.modules import loss
from torch.optim import Adam
from torch.utils.data import DataLoader
from utils.evaluation import mae, mse, rmse
from utils.model_util import freeze_random

from .model import FedNeuMFModel
"""
RESULT FedNCF:
0609
01 分布 0.0005

2.5%
Epoch:40 mae:24.383689880371094,mse:4902.00830078125,rmse:70.01434326171875
Epoch:40 mae:0.5325819253921509,mse:2.3127601146698,rmse:1.5207761526107788


7.5%
Epoch:220 mae:0.4605955481529236,mse:1.9085830450057983,rmse:1.3815147876739502
Epoch:250 mae:17.417064666748047,mse:2790.504638671875,rmse:52.825225830078125

"""
import os
# freeze_random()  # 冻结随机数 保证结果一致
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
density = 0.075
type_ = "tp"
rt_data = MatrixDataset(type_)
train_data, test_data = rt_data.split_train_test(density)

train_dataset = ToTorchDataset(train_data)
test_dataset = ToTorchDataset(test_data)

test_dataloader = DataLoader(test_dataset, batch_size=2048)

lr = 0.0005
epochs = 300
# loss_fn = nn.SmoothL1Loss()
loss_fn = nn.L1Loss()

neumf = FedNeuMFModel(
    train_data,
    loss_fn,
    rt_data.row_n,
    rt_data.col_n,
    dim=8,
    layers=[64, 32, 8],
)

neumf.fit(epochs, lr, test_dataloader)
# y, y_pred = neumf.predict(
#     test_dataloader, False,
# )
# mae_ = mae(y, y_pred)
# mse_ = mse(y, y_pred)
# rmse_ = rmse(y, y_pred)

# print(f"Density:{density},type:{type_},mae:{mae_},mse:{mse_},rmse:{rmse_}")
