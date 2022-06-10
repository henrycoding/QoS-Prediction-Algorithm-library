import os

import torch
from data import MatrixDataset, ToTorchDataset
from models.NCF.model import NCF, NCFModel
from root import ROOT, absolute
from torch import nn, optim
from torch.nn.modules import loss
from torch.optim import Adam
from torch.utils.data import DataLoader
from utils.evaluation import mae, mse, rmse
# 冻结随机数
from utils.model_util import freeze_random
# 日志
from utils.mylogger import TNLog
"""
RESULT NeuMF:
Density:0.05, type:rt, mae:0.6073, mse:3.6153, rmse:1.9014
Density:0.10, type:rt, mae:0.5820, mse:3.5959, rmse:1.8963
Density:0.15, type:rt, mae:0.5713, mse:3.5806, rmse:1.8922
Density:0.20, type:rt, mae:0.5651, mse:3.5951, rmse:1.8961

0609
[64,32,8]

0 1分布 0.001 定稿✅
01分布 0.001
Epoch:180 mae:0.47287681698799133,mse:2.0685651302337646,rmse:1.4382506608963013
Epoch:130 mae:0.3863808214664459,mse:1.7274330854415894,rmse:1.3143185377120972
 Epoch:100 mae:0.3627842962741852,mse:1.6989020109176636,rmse:1.3034193515777588
Epoch:100 mae:0.352378249168396,mse:1.6250346899032593,rmse:1.274768471717834

0 1 分布 0.001
Epoch:150 mae:19.348459243774414,mse:3139.612060546875,rmse:56.03224182128906
mae:14.816617012023926,mse:2395.664306640625,rmse:48.94552230834961
Epoch:130 mae:13.307035446166992,mse:2057.2890625,rmse:45.35734939575195
mae:12.846741676330566,mse:2020.776611328125,rmse:44.95304870605469


"""

# freeze_random()  # 冻结随机数 保证结果一致

# logger = TNLog('NeuMF')
# logger.initial_logger()

# for density in [0.05, 0.1, 0.15, 0.2]:
density = 0.05
type_ = "tp"
rt_data = MatrixDataset(type_)
train_data, test_data = rt_data.split_train_test(density)

train_dataset = ToTorchDataset(train_data)
test_dataset = ToTorchDataset(test_data)
train_dataloader = DataLoader(train_dataset, batch_size=512)
test_dataloader = DataLoader(test_dataset, batch_size=2048)

lr = 0.0005
epochs = 500
dim = 8

loss_fn = nn.L1Loss()

ncf = NCFModel(loss_fn,rt_data.row_n, rt_data.col_n,dim,[64,32,8])
opt = Adam(ncf.parameters(), lr=lr)

ncf.fit(train_dataloader, epochs, opt, eval_loader=test_dataloader,
            save_filename=f"Density_{density}")

y, y_pred = ncf.predict(test_dataloader, True)
mae_ = mae(y, y_pred)
mse_ = mse(y, y_pred)
rmse_ = rmse(y, y_pred)

ncf.logger.info(
    f"Density:{density:.2f}, type:{type_}, mae:{mae_:.4f}, mse:{mse_:.4f}, rmse:{rmse_:.4f}")
