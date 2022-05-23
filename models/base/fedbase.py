import copy
from collections import OrderedDict, defaultdict
from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm

from utils.model_util import use_optimizer

from .utils import train_mult_epochs_with_dataloader,freeze_specific_params,unfreeze_params


class ClientBase(object):
    def __init__(self, device, model, personalized=False) -> None:
        self.device = device
        self.model = model
        self.data_loader = None
        self.test_data_loader = None
        self.is_personalized = personalized
        super().__init__()



    def fit(self,
            params,
            loss_fn,
            optimizer: str,
            lr,
            epochs=5,
            header_epoch=None):

        # 用本地模型的参数替换服务端的模型
        if self.is_personalized:
            for name, param in self.model.named_parameters():
                if self.model.personal_layer in name:
                    params[name] = param

        self.model.load_state_dict(params)
        self.model.to(self.device)
        opt = use_optimizer(self.model, optimizer, lr)

        if self.is_personalized and header_epoch is not None:
            freeze_specific_params(self.model,self.model.personal_layer)
            train_mult_epochs_with_dataloader(
                epochs,
                model=self.model,
                device=self.device,
                dataloader=self.data_loader,
                opt=opt,
                loss_fn=loss_fn
            )
            unfreeze_params(self.model)

        loss, lis = train_mult_epochs_with_dataloader(
            epochs,
            model=self.model,
            device=self.device,
            dataloader=self.data_loader,
            opt=opt,
            loss_fn=loss_fn
        
        )
        self.loss_list = [*lis]
        return self.model.state_dict(), round(loss, 4)

    def predict(self, params):

        # 用本地模型的参数替换服务端的模型
        if self.is_personalized:
            for name, param in self.model.named_parameters():
                if self.model.personal_layer in name:
                    params[name] = param

        self.model.load_state_dict(params)
        self.model.to(self.device)
        self.model.eval()
        with torch.no_grad():
            y_pred_list = []
            y_list = []
            # for batch_id, batch in tqdm(enumerate(test_loader)):
            for batch_id, batch in enumerate(self.test_data_loader):
                user, item, rate = batch[0].to(self.device), batch[1].to(
                    self.device), batch[2].to(self.device)
                y_pred = self.model(user, item).squeeze()
                y_real = rate.reshape(-1, 1)

                if len(y_pred.shape) == 0:  # 64一batch导致变成了标量
                    y_pred = y_pred.unsqueeze(dim=0)
                if len(y_real.shape) == 0:
                    y_real = y_real.unsqueeze(dim=0)

                y_pred_list.append(y_pred)
                y_list.append(y_real)

            y_pred_list = torch.cat(y_pred_list).cpu().numpy()
            y_list = torch.cat(y_list).cpu().numpy()

            return y_list, y_pred_list


class ClientsBase(object):
    """多client 的虚拟管理节点
    """
    def __init__(self, triad, model, device) -> None:
        super().__init__()
        self.triad = triad
        self.model = model
        self.device = device
        self.clients_map = {}  # 存储每个client的数据集

        self._get_clients()

    def sample_clients(self, fraction):
        """Select some fraction of all clients."""
        num_clients = len(self.clients_map)
        num_sampled_clients = max(int(fraction * num_clients), 1)
        sampled_client_indices = sorted(
            np.random.choice(a=[k for k, v in self.clients_map.items()],
                             size=num_sampled_clients,
                             replace=False).tolist())
        return sampled_client_indices

    def _get_clients(self):
        raise NotImplementedError

    def __len__(self):
        return len(self.clients_map)

    def __iter__(self):
        for item in self.clients_map.items():
            yield item

    def __getitem__(self, uid):
        return self.clients_map[uid]


class ServerBase(object):
    def __init__(self) -> None:
        super().__init__()

    def upgrade_wich_cefficients(self, params: List[Dict], coefficients: Dict,
                                 personal_layer_name: str):
        """使用加权平均对参数进行更新

        Args:
            params : 模型参数
            coefficients : 加权平均的系数

        personal_layer_name 不聚合这些参数
        """

        o = OrderedDict()
        if len(params) != 0:
            # 获得不同的键
            for k, v in params[0].items():
                # 个性化层不做聚合
                # if personal_layer_name in k:
                #     print(k)
                #     continue
                # 实际操作个性化层也做聚合,在客户端再覆盖
                for it, param in enumerate(params):

                    if it == 0:
                        o[k] = coefficients[it] * param[k]
                    else:
                        o[k] += coefficients[it] * param[k]
            self.params = o

    def upgrade_average(self, params: List[Dict], personal_layer_name: str):
        o = OrderedDict()
        if len(params) != 0:
            for k, v in params[0].items():
                if personal_layer_name in k:
                    continue
                o[k] = sum([i[k] for i in params]) / len(params)
            self.params = o


class FedModelBase(object):
    def update_selected_clients(self, sampled_client_indices, lr, s_params):
        """使用 client.fit 函数来训练被选择的client
        """
        collector = []
        client_loss = []
        selected_total_size = 0  # client数据集总数

        for uid in tqdm(sampled_client_indices,
                        desc="Client training",
                        colour="green",
                        ncols=80):
            s_params, loss = self.clients[uid].fit(s_params, self.loss_fn,
                                                   self.optimizer, lr)
            collector.append(s_params)
            client_loss.append(loss)
            selected_total_size += self.clients[uid].n_item
        return collector, client_loss, selected_total_size

    def evaluation_selected_clients(self, client_indices, params):
        y_list = []
        y_pred_list = []
        for uid in tqdm(client_indices,
                        desc="Client Evaluation",
                        colour="green",
                        ncols=80):
            y, y_pred = self.clients[uid].predict(params)
            y_list.extend(y)
            y_pred_list.extend(y_pred)
        return y_list, y_pred_list

    def _check(self, iterator):
        assert abs(sum(iterator) - 1) <= 1e-4
