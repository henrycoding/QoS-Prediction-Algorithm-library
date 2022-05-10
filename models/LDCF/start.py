import time
from collections import defaultdict

import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import InfoDataset, MatrixDataset, ToTorchDataset
from models.LDCF.config import get_cfg_defaults
from models.LDCF.model import FedLDCFModel
from utils.evaluation import mae, mse, rmse
from utils.model_util import freeze_random, split_d_triad


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
        r.append([[uid, iid, rate], [u, i, rate]]) if is_dtriad else r.append(
            [u, i, rate])
    return r


# enable_columns = ["[Latitude]", "[Longitude]"]
# u_info = InfoDataset("user", enable_columns)
# i_info = InfoDataset("service", enable_columns)
# type_ = 'rt'
# md = MatrixDataset(type_)
# train, test = md.split_train_test(0.05)
# train_data = data_preprocess(train, u_info, i_info, True)
# test_data = data_preprocess(test, u_info, i_info, True)
# train_dataset = ToTorchDataset(train_data)
# test_dataset = ToTorchDataset(test_data)
# train_dataloader = DataLoader(train, batch_size=128)
# test_dataloader = DataLoader(test_dataset, batch_size=2048)


freeze_random()  # 冻结随机数 保证结果一致
cfg = get_cfg_defaults()

u_enable_columns = ["[User ID]", "[Latitude]", "[Longitude]"]
i_enable_columns = ["[Service ID]", "[Latitude]", "[Longitude]"]
u_info = InfoDataset("user", u_enable_columns)
i_info = InfoDataset("service", i_enable_columns)
date = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
density_list = cfg.TRAIN.DENSITY_LIST
type_ = cfg.TRAIN.DATA_TYPE
rt_data = MatrixDataset(type_)
batch_size = cfg.TRAIN.BATCH_SIZE
lr = cfg.TRAIN.OPTIMIZER.LR
epochs = cfg.TRAIN.NUM_EPOCHS

cfg.defrost()
cfg.MODEL.DATE = date
cfg.TRAIN.NUM_USERS = rt_data.row_n
cfg.TRAIN.NUM_ITEMS = rt_data.col_n
cfg.TRAIN.NUM_USERS_AC = max(u_info.feature2num.values())
cfg.TRAIN.NUM_ITEM_AC = max(i_info.feature2num.values())
cfg.freeze()

result = defaultdict(list)
for density in density_list:
    train, test = rt_data.split_train_test(density)
    train_data = data_preprocess(train, u_info, i_info, True)
    test_data = data_preprocess(test, u_info, i_info, True)
    testDataset, p_test = split_d_triad(test_data)
    # train_dataset = ToTorchDataset(train_data)
    # test_dataset = ToTorchDataset(test_data)
    # train_dataloader = DataLoader(p_triad, batch_size=batch_size, drop_last=True)
    test_dataloader = DataLoader(ToTorchDataset(p_test), batch_size=batch_size, drop_last=True)

    ldcf = FedLDCFModel(train_data, cfg)

    ldcf.fit(epochs, lr, test_dataloader, date, density)

    y, y_pred = ldcf.predict(
        test_dataloader, False,
    )
    mae_ = mae(y, y_pred)
    mse_ = mse(y, y_pred)
    rmse_ = rmse(y, y_pred)
    result[density].extend([mae_, mse_, rmse_])
    print(f"Density:{density},type:{type_},mae:{mae_},mse:{mse_},rmse:{rmse_}")

result_show = pd.DataFrame(result, index=['MAE', 'MSE', 'RMSE'])
print(result_show)
