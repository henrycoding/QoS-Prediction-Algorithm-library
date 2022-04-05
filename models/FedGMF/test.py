import torch
from data import MatrixDataset, ToTorchDataset
from root import absolute
from torch import nn, optim
from torch.nn.modules import loss
from torch.optim import Adam
from torch.utils.data import DataLoader
from utils.evaluation import mae, mse, rmse
from utils.model_util import freeze_random

from .model import FedGMF, FedGMFModel
"""
RESULT FedGMF:
5% Epoch:130 mae:0.6076180934906006,mse:2.7448225021362305,rmse:1.6567505598068237
10% Epoch:200 mae:0.5711803436279297,mse:2.7105050086975098,rmse:1.6463611125946045
15% Epoch:130 mae:0.5654447674751282,mse:2.530756711959839,rmse:1.5908352136611938
20% Epoch:120 mae:0.5369598269462585,mse:2.4346539974212646,rmse:1.560337781906128

5% Epoch:90 mae:29.87485694885254,mse:5886.099609375,rmse:76.72091674804688
10% Epoch:340 mae:22.7558536529541,mse:4222.40478515625,rmse:64.98003387451172
15%    mae:21.49823760986328,mse:4201.45458984375,rmse:64.8186264038086
20%   mae:21.878353118896484,mse:4757.3271484375,rmse:68.97338104248047
"""

freeze_random()  # 冻结随机数 保证结果一致

density = 0.2


type_ = "tp"
rt_data = MatrixDataset(type_)
train_data, test_data = rt_data.split_train_test(density)

train_dataset = ToTorchDataset(train_data)
test_dataset = ToTorchDataset(test_data)

test_dataloader = DataLoader(test_dataset, batch_size=2048)

lr = 0.001
epochs = 1000
# loss_fn = nn.SmoothL1Loss()
loss_fn = nn.L1Loss()

dim = 12

gmf = FedGMFModel(train_data,
                    loss_fn,
                    rt_data.row_n,
                    rt_data.col_n,
                    dim=dim,
                    layer=[12,64,32])

gmf.fit(epochs, lr, test_dataloader,1)
y, y_pred = gmf.predict(
    test_dataloader, False,
    "/Users/wenzhuo/Desktop/研究生/科研/QoS预测实验代码/SCDM/output/FedMLPModel/loss_0.5389.ckpt"
)
mae_ = mae(y, y_pred)
mse_ = mse(y, y_pred)
rmse_ = rmse(y, y_pred)

print(f"Density:{density},type:{type_},mae:{mae_},mse:{mse_},rmse:{rmse_}")
