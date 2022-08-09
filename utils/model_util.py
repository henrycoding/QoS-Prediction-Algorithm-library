import copy
import os
import random
import shutil
import time

import numpy as np
import torch
from tqdm import tqdm

from root import absolute
from torch import nn

from torch.utils.tensorboard import SummaryWriter
from tensorboard import program

# config
from yacs.config import CfgNode

# loading dataset
from data import MatrixDataset, ToTorchDataset, normalization, InfoDataset
from torch.utils.data import DataLoader

# evaluation indicator
from utils.evaluation import mae, mse, rmse

# Store and display model result
import pandas as pd
from collections import defaultdict

from sklearn.preprocessing import normalize

"""
    Some handy functions for model training ...
"""


# 公用模型测试框架
class ModelTest:
    def __init__(self, model_class, config: CfgNode) -> None:
        self.config = config

        # model
        self.model_class = model_class  # the class of the model
        self.engine = None

        # running time
        self.date = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())

        # density
        self.density = None
        try:
            density_list = self.config.TRAIN.DENSITY_LIST
        except Exception:
            raise Exception("The 'TRAIN.DENSITY_LIST' is not provided in the configuration file!")
        else:
            self.density_list = density_list

        # data type
        try:
            data_type = self.config.TRAIN.DATA_TYPE
        except Exception:
            raise Exception("The 'TRAIN.DATA_TYPE' is not provided in the configuration file!")
        else:
            self.data_type = data_type

        # batch_size
        try:
            batch_size = self.config.TRAIN.BATCH_SIZE
        except Exception:
            raise Exception("The 'TRAIN.BATCH_SIZE' is not provided in the configuration file!")
        else:
            self.batch_size = batch_size

        # store result
        self.result = defaultdict(list)
        self.result_show = None

        # TensorBoard writer
        self.writer = None

    def run(self, is_fed=False) -> None:
        freeze_random()  # frozen random number 保证可重复
        # TODO
        self.writer = TensorBoardTool(self.config).run()

        for density in self.density_list:
            self.density = density
            rt_data, train_dataloader, test_dataloader = data_loading(self.data_type, density, self.batch_size,
                                                                      is_fed=is_fed)
            num_users = rt_data.row_n
            num_items = rt_data.col_n

            self.config.defrost()
            self.config.MODEL.DATE = self.date
            self.config.TRAIN.NUM_USERS = num_users
            self.config.TRAIN.NUM_ITEMS = num_items
            self.config.TRAIN.DENSITY = density
            self.config.freeze()
            self.engine = self.model_class(self.config, self.writer)  # 实例化ModelBase对象

            self._fit(train_dataloader, test_dataloader)
            self._predict(test_dataloader)

        self.result_show = pd.DataFrame(self.result, index=['MAE', 'MSE', 'RMSE'])
        print(
            'experimental results of service QoS prediction in ' +
            'response time' if self.data_type == 'rt' else 'throughput')
        print(self.result_show)

    def _fit(self, train_dataloader, test_dataloader) -> None:
        self.engine.fit(train_dataloader, test_dataloader)

    def _predict(self, test_dataloader):
        y, y_pred = self.engine.predict(test_dataloader)

        mae_ = mae(y, y_pred)
        mse_ = mse(y, y_pred)
        rmse_ = rmse(y, y_pred)

        self.result[self.density].extend([mae_, mse_, rmse_])
        self.engine.logger.info(
            f"Density:{self.density:.2f}, type:{self.data_type}, mae:{mae_:.4f}, mse:{mse_:.4f}, rmse:{rmse_:.4f}")


class TensorBoardTool:
    """
    Tensorboard automatic deployment
    """

    def __init__(self, config: CfgNode) -> None:
        self.config = config

        # model name
        try:
            name = self.config.MODEL.NAME
        except Exception:
            raise Exception("The 'MODEL.NAME' is not provided in the configuration file!")
        else:
            self.model_name = name

        # model directory
        try:
            directory = self.config.MODEL.DIR
        except Exception:
            raise Exception("The 'MODEL.DIR' is not provided in the configuration file!")
        else:
            self.model_dir = directory

    def _move_tb_history_file(self) -> None:
        """
        Move the historical log file to the 'output' directory
        """
        source_dir = absolute(f"{self.model_dir}/runs")
        save_dir = absolute(f"output/{self.model_name}/runs")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        if os.path.exists(source_dir):
            files = os.listdir(source_dir)
            for f in files:
                shutil.move(os.path.join(source_dir, f), save_dir)

    def run(self, clear: bool = True):
        """
        Start TensorBoard

        Args:
            clear (bool): whether to move the historical log file.
                If true, only the results of this run are displayed

        Returns:
            writer (): TensorBoard's writer
        """
        if clear:
            self._move_tb_history_file()

        writer = SummaryWriter()
        tensorboard = program.TensorBoard()
        tensorboard.configure(argv=[None, '--logdir', absolute(f'{self.model_dir}/runs')])
        tensorboard.launch()
        print('TensorBoard running at ' + 'http://localhost:6006/')
        return writer

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


def data_loading(data_type: str, density: float, batch_size=64, is_fed=False):
    """
    Read the raw data, devide train sets and test sets, process them into dataloader

    Args:
        data_type (): dataset type(response time / throughput)
        density (): density of the training data
        batch_size (): batch size

    Returns:
        row_data (): row data
        train_dataloader (): dataloader for train
        test_dataloader (): dataloader for test
    """
    row_data = MatrixDataset(data_type)

    train_data, test_data = row_data.split_train_test(density)

    # TODO 归一化处理
    # train_data[:, -1] = normalization(train_data[:, -1])

    if not is_fed:
        train_dataset = ToTorchDataset(train_data)
        test_dataset = ToTorchDataset(test_data)

        train_dataloader = DataLoader(train_dataset, batch_size)
        test_dataloader = DataLoader(test_dataset, batch_size)
        return row_data, train_dataloader, test_dataloader
    else:
        u_enable_columns = ["[User ID]", "[Latitude]", "[Longitude]"]
        i_enable_columns = ["[Service ID]", "[Latitude]", "[Longitude]"]
        u_info = InfoDataset("user", u_enable_columns)
        i_info = InfoDataset("service", i_enable_columns)
        # train, test = rt_data.split_train_test(density)
        train_data_preprocess = data_preprocess(train_data, u_info, i_info, True)
        test_data_preprocess = data_preprocess(test_data, u_info, i_info, True)
        train_dataset, p_rain = split_d_triad(train_data_preprocess)
        test_dataset, p_test = split_d_triad(test_data_preprocess)
        train_dataloader = DataLoader(ToTorchDataset(p_rain), batch_size=batch_size, drop_last=True)
        test_dataloader = DataLoader(ToTorchDataset(p_test), batch_size=batch_size, drop_last=True)
        return row_data, train_dataloader, test_dataloader


def use_loss_fn(params: CfgNode):
    """
    Select loss function according to the configuration file
    """
    try:
        type_ = params.TRAIN.LOSS_FN.TYPE
    except Exception:
        raise Exception("The 'TRAIN.LOSS_FN.TYPE' is not provided in the configuration file!")

    if type_ == 'L1':
        return torch.nn.L1Loss()
    elif type_ == 'SmoothL1':
        return torch.nn.SmoothL1Loss()
    elif type_ == "Huber":
        return torch.nn.HuberLoss()
    else:
        raise NotImplementedError('loss function not support')


def use_optimizer(network, params: CfgNode):
    """
    Select optimizer according to the configuration file
    """
    try:
        type_ = params.TRAIN.OPTIMIZER.TYPE
    except Exception:
        raise Exception("The 'TRAIN.OPTIMIZER.TYPE' is not provided in the configuration file!")

    config = params.TRAIN.OPTIMIZER
    if type_ == 'SGD':
        return torch.optim.SGD(network.parameters(),
                               lr=config.get('LR', 1e-3),
                               momentum=config.get('MOMENTUM', 0),
                               weight_decay=config.get('WEIGHT_DECAY', 0))
    elif type_ == 'Adam':
        return torch.optim.Adam(network.parameters(),
                                lr=config.get('LR', 1e-3),
                                weight_decay=config.get('WEIGHT_DECAY', 0))
    elif type_ == 'RMSprop':
        return torch.optim.RMSprop(network.parameters(),
                                   lr=config.get('LR', 1e-3),
                                   alpha=config.get('ALPHA', 0.99),
                                   momentum=config.get('MOMENTUM', 0))


def save_checkpoint(state,
                    is_best,
                    save_dirname="output",
                    save_filename="best_model.ckpt"):
    """Save checkpoint if a new best is achieved"""
    if not os.path.isdir(absolute(save_dirname)):
        os.makedirs(absolute(save_dirname))
    file_path = absolute(f"{save_dirname}/{save_filename}")
    if is_best:
        print("\n=> Saving a new best")
        print(file_path)
        torch.save(state, file_path)  # save checkpoint
    else:
        print("=> Validation Accuracy did not improve")


def load_checkpoint(file_path: str, device=None):
    """Loads torch model from checkpoint file.
    Args:
        file_path (str): Path to checkpoint directory or filename
    """
    if not os.path.exists(file_path):
        raise Exception("ckpt file doesn't exist")
    ckpt = torch.load(file_path, map_location=device)
    print(' [*] Loading checkpoint from %s succeed!' % file_path)
    return ckpt


def get_device(config: CfgNode):
    try:
        use_gpu = config.SYSTEM.USE_GPU
    except:
        return 'cpu'
    else:
        if use_gpu and torch.cuda.is_available():
            return 'cuda'
        else:
            return 'cpu'


def use_cuda(enabled, device_id=0):
    if enabled:
        assert torch.cuda.is_available(), 'CUDA is not available'
        torch.cuda.set_device(device_id)


def freeze_random(seed=2021):
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def triad_to_matrix(triad, nan_symbol=-1):
    """三元组转矩阵

    Args:
        triad : 三元组
        nan_symbol : 非零数据的表示方法. Defaults to -1.

    """
    # 注意下标应该为int
    if not isinstance(triad, np.ndarray):
        triad = np.array(triad)
    x_max = triad[:, 0].max().astype(int)  # 用户数量
    y_max = triad[:, 1].max().astype(int)  # 项目数量
    matrix = np.full((x_max + 1, y_max + 1), nan_symbol,
                     dtype=triad.dtype)  # 初始化QoS矩阵
    matrix[triad[:, 0].astype(int),
           triad[:, 1].astype(int)] = triad[:, 2]  # 将评分值放到QoS矩阵的对应位置中
    return matrix


def split_d_triad(d_triad):
    l = np.array(d_triad, dtype=np.object)
    return np.array(l[:, 0].tolist()), l[:, 1].tolist()


def nonzero_user_mean(matrix, nan_symbol):
    """快速计算一个矩阵的行均值
    """
    m = copy.deepcopy(matrix)
    m[matrix == nan_symbol] = 0
    t = (m != 0).sum(axis=-1)  # 每行非0元素的个数
    res = (m.sum(axis=-1) / t).squeeze()
    res[np.isnan(res)] = 0
    return res


def nonzero_item_mean(matrix, nan_symbol):
    """快速计算一个矩阵的列均值
    """
    return nonzero_user_mean(matrix.T, nan_symbol)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


#########################
# Weight initialization #
#########################
def init_weights(model, init_type, init_gain):
    """Function for initializing network weights.

    Args:
        model: A torch.nn instance to be initialized.
        init_type: Name of an initialization method (normal | xavier | kaiming | orthogonal).
        init_gain: Scaling factor for (normal | xavier | orthogonal).

    Reference:
        https://github.com/DS3Lab/forest-prediction/blob/master/pix2pix/models/networks.py
    """

    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1
                                     or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            else:
                raise NotImplementedError(
                    f'[ERROR] ...initialization method [{init_type}] is not implemented!'
                )
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

        elif classname.find('BatchNorm2d') != -1 or classname.find(
                'InstanceNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    model.apply(init_func)


def init_net(model, init_type, init_gain, gpu_ids):
    """Function for initializing network weights.

    Args:
        model: A torch.nn.Module to be initialized
        init_type: Name of an initialization method (normal | xavier | kaiming | orthogonal)l
        init_gain: Scaling factor for (normal | xavier | orthogonal).
        gpu_ids: List or int indicating which GPU(s) the network runs on. (e.g., [0, 1, 2], 0)

    Returns:
        An initialized torch.nn.Module instance.
    """
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        model.to(gpu_ids[0])
        model = nn.DataParallel(model, gpu_ids)
    init_weights(model, init_type, init_gain)
    return model


if __name__ == "__main__":
    d_triad = [[[1, 2, 3.2], [[1, 1], [2, 2], 3.2]],
               [[1, 2, 3.2], [[1, 1], [2, 2], 3.2]],
               [[1, 2, 3.2], [[1, 1], [2, 2], 3.2]],
               [[1, 2, 3.2], [[1, 1], [2, 2], 3.2]]]
    a, b = split_d_triad(d_triad)
    t2m = triad_to_matrix(a)
    print(t2m)
    print(nonzero_user_mean(t2m, -1))
