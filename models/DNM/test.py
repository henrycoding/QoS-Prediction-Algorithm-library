from torch.testing._internal.distributed.rpc.examples.parameter_server_test import batch_size
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
import torch
from torch import nn

from data import InfoDataset, MatrixDataset, ToTorchDataset
from models.DNM.model import DNMModel

desnity = 0.05

u_enable_columns = ["[User ID]", "[Country]", "[AS]"]
i_enable_columns = ["[Service ID]", "[Country]", "[AS]"]


def data_preprocess(triad,
                    u_info_obj: InfoDataset,
                    i_info_obj: InfoDataset,
                    is_dtriad=False):
    """生成d_triad [[triad],[p_triad]]
    """
    r = []
    for row in tqdm(triad, desc="Gen d_triad"):
        uid, iid, rate = int(row[0]), int(row[1]), float(row[2])
        u = u_info_obj.query(uid)
        i = i_info_obj.query(iid)
        r.append([u, i, rate])
    return r


md = MatrixDataset()
u_info = InfoDataset("user", u_enable_columns)
i_info = InfoDataset("service", i_enable_columns)
train, test = md.split_train_test(desnity)

loss_fn = nn.L1Loss()

train_data = data_preprocess(train, u_info, i_info)
test_data = data_preprocess(test, u_info, i_info)
train_dataset = ToTorchDataset(train_data)
test_dataset = ToTorchDataset(test_data)
train_dataloader = DataLoader(train_dataset, batch_size=128)
test_dataloader = DataLoader(test_dataset, batch_size=128)

model = DNMModel()
opt = Adam(model.parameters(), lr=0.001)

model.fit(train_dataloader, test_dataloader)
