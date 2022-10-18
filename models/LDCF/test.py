import torch
from data import MatrixDataset, ToTorchDataset
from models.LDCF.model import LDCFModel
from root import absolute
from torch import nn, optim
from torch.nn.modules import loss
from torch.optim import Adam
from torch.utils.data import DataLoader
from utils.evaluation import mae, mse, rmse
from utils.model_util import count_parameters, freeze_random
from data import InfoDataset, MatrixDataset, ToTorchDataset
from tqdm import tqdm
"""
RESULT LDCF:
"""

freeze_random()  # 冻结随机数 保证结果一致



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



for density in [0.05, 0.1, 0.15, 0.2]:

    type_ = "rt"
    md = MatrixDataset(type_)
    u_info = InfoDataset("user", u_enable_columns)
    i_info = InfoDataset("service", i_enable_columns)
    train, test = md.split_train_test(density)
    user_params = {
        "type_": "cat",  # embedding层整合方式 stack or cat
        "embedding_nums": u_info.embedding_nums,  # 每个要embedding的特征的总个数
        "embedding_dims": [16, 8, 8],
    }

    item_params = {
        "type_": "cat",  # embedding层整合方式 stack or cat
        "embedding_nums": i_info.embedding_nums,  # 每个要embedding的特征的总个数
        "embedding_dims": [16, 8, 8],
    }

    train_data = data_preprocess(train, u_info, i_info)
    test_data = data_preprocess(test, u_info, i_info)
    train_dataset = ToTorchDataset(train_data)
    test_dataset = ToTorchDataset(test_data)
    train_dataloader = DataLoader(train_dataset, batch_size=128)
    test_dataloader = DataLoader(test_dataset, batch_size=2048)


    lr = 0.0001
    epochs = 100

    # loss_fn = nn.L1Loss()
    loss_fn = nn.HuberLoss()
    ldcf = LDCFModel(loss_fn,user_params,item_params,[64,32,16,8])


    opt = Adam(ldcf.parameters(), lr=lr)

    print(f"模型参数:", count_parameters(ldcf))
    ldcf.fit(train_dataloader, epochs, opt, eval_loader=test_dataloader)

    # y, y_pred = ldcf.predict(test_dataloader)

    # mae_ = mae(y, y_pred)
    # mse_ = mse(y, y_pred)
    # rmse_ = rmse(y, y_pred)

    # print(f"Density:{density},type:{type_},mae:{mae_},mse:{mse_},rmse:{rmse_}")
