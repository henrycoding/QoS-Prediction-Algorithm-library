import copy
from collections import OrderedDict, defaultdict

import numpy as np
import torch
from data import ToTorchDataset
from models.base import ClientBase, ClientsBase
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.model_util import (nonzero_user_mean, split_d_triad,
                              triad_to_matrix, use_optimizer)


class Client(ClientBase):
    """客户端实体
    """
    def __init__(self,
                 triad,
                 test_triad,
                 uid,
                 model,
                 device,
                 batch_size=32) -> None:
        """客户端调用fit进行训练

        Args:
            triad: 三元组
            batch_size : local 训练的bs, 默认10, -1表示
        """

        super().__init__(device, model)
        self.triad = triad
        self.test_triad = test_triad
        self.uid = uid
        self.loss_list = []
        self.n_item = len(triad)
        self.batch_size = batch_size if batch_size != -1 else self.n_item
        self.data_loader = DataLoader(ToTorchDataset(self.triad),
                                      batch_size=self.batch_size)

        # 每一个节点要有测试集和训练集
        self.test_data_loader = DataLoader(ToTorchDataset(self.test_triad),
                                           batch_size=len(test_triad),
                                           drop_last=True)

    def fit(self, params, loss_fn, optimizer: str, lr, epochs=5):
        return super().fit(params, loss_fn, optimizer, lr, epochs=epochs)

    def predict(self, params):
        return super().predict(params)

    def __repr__(self) -> str:
        return f"Client(uid={self.uid})"


class Clients(ClientsBase):
    """多client 的虚拟管理节点
    """
    def __init__(self,
                 triad,
                 test_triad,
                 model,
                 device,
                 batch_size=-1) -> None:
        self.client_nums_map = {}
        self.batch_size = batch_size
        self.test_triad = test_triad

        super().__init__(triad, model, device)

        self._get_clients()

    def _get_clients(self):
        train_data_dic = defaultdict(list)
        for triad_row in self.triad:
            uid, iid, rate = int(triad_row[0]), int(triad_row[1]), float(
                triad_row[2])
            train_data_dic[uid].append(triad_row)

        test_data_dic = defaultdict(list)
        for triad_row in self.test_triad:
            uid, iid, rate = int(triad_row[0]), int(triad_row[1]), float(
                triad_row[2])
            test_data_dic[uid].append(triad_row)

        for uid, rows in tqdm(train_data_dic.items(),
                              desc="Building clients...",
                              ncols=80):
            self.clients_map[uid] = Client(rows,
                                           test_data_dic[uid],
                                           uid,
                                           copy.deepcopy(self.model),
                                           self.device,
                                           batch_size=self.batch_size)
            self.client_nums_map[uid] = len(rows)
        print(f"Clients Nums:{len(self.clients_map)}")
        print(f"Nums for client:", self.client_nums_map)
