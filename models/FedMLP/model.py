from typing import OrderedDict

import torch
from models.base import FedModelBase
from torch import nn
from tqdm import tqdm
from utils.evaluation import mae, mse, rmse
from utils.model_util import load_checkpoint, save_checkpoint
from utils.mylogger import TNLog

from .client import Clients
from .server import Server


class Linear(nn.Module):
    def __init__(self,
                 in_size,
                 out_size,
                 activation,
                 personal_layer_name=None):
        super().__init__()
        if personal_layer_name is None:
            self.fc_layer = nn.Sequential(nn.Linear(in_size, out_size),
                                          activation())
        else:
            self.fc_layer = nn.Sequential(
                OrderedDict({
                    personal_layer_name: nn.Linear(in_size, out_size),
                    "activation": activation()
                }))

    def forward(self, x):
        x = self.fc_layer(x)
        return x


class FedMLP2(nn.Module):
    def __init__(self,
                 n_user,
                 n_item,
                 dim=8,
                 layers=[8, 32, 64],
                 personal_layers=[128, 64, 1]) -> None:
        """
        Args:
            n_user ([type]): 用户数量
            n_item ([type]): 物品数量
            dim ([type]): 特征空间的维度
            layers (list, optional): 多层感知机的层数. Defaults to [16,32,16,8].
            output_dim (int, optional): 最后输出的维度. Defaults to 1.
        """
        super(FedMLP2, self).__init__()
        self.num_users = n_user
        self.num_items = n_item
        self.latent_dim = dim

        self.personal_layer = "my_layer"

        self.embedding_user = nn.Embedding(num_embeddings=self.num_users,
                                           embedding_dim=self.latent_dim)
        self.embedding_item = nn.Embedding(num_embeddings=self.num_items,
                                           embedding_dim=self.latent_dim)

        self.user_encoder = nn.Sequential(*[
            Linear(in_size, out_size, nn.ReLU)
            for (in_size, out_size) in zip(layers, layers[1:])
        ])

        self.item_encoder = nn.Sequential(*[
            Linear(in_size, out_size, nn.ReLU)
            for (in_size, out_size) in zip(layers, layers[1:])
        ])

        self.my_layer = nn.Sequential(*[
            Linear(in_size, out_size, nn.ReLU)
            for (in_size,
                 out_size) in zip(personal_layers, personal_layers[1:])
        ])

    def forward(self, user_idx, item_idx):
        user_embedding = self.embedding_user(user_idx)
        item_embedding = self.embedding_item(item_idx)
        user_x = self.user_encoder(user_embedding)
        item_x = self.item_encoder(item_embedding)
        x = torch.cat([user_x, item_x], dim=-1)
        x = self.my_layer(x)
        return x


class FedMLP(nn.Module):
    def __init__(self,
                 n_user,
                 n_item,
                 dim,
                 layers=[32, 16, 8],
                 output_dim=1) -> None:
        """
        Args:
            n_user ([type]): 用户数量
            n_item ([type]): 物品数量
            dim ([type]): 特征空间的维度
            layers (list, optional): 多层感知机的层数. Defaults to [16,32,16,8].
            output_dim (int, optional): 最后输出的维度. Defaults to 1.
        """
        super(FedMLP, self).__init__()
        self.num_users = n_user
        self.num_items = n_item
        self.latent_dim = dim

        self.embedding_user = nn.Embedding(num_embeddings=self.num_users,
                                           embedding_dim=self.latent_dim)
        self.embedding_item = nn.Embedding(num_embeddings=self.num_items,
                                           embedding_dim=self.latent_dim)

        self.fc_layers = nn.ModuleList()
        # MLP的第一层为latent vec的cat
        self.fc_layers.append(nn.Linear(self.latent_dim * 2, layers[0]))

        for in_size, out_size in zip(layers, layers[1:]):
            self.fc_layers.append(nn.Linear(in_size, out_size))

        self.my_layer = nn.Sequential(
            OrderedDict({"my_layer": nn.Linear(layers[-1], layers[-1])}))

        self.fc_output = nn.Linear(layers[-1], output_dim)

    def forward(self, user_idx, item_idx):
        user_embedding = self.embedding_user(user_idx)
        item_embedding = self.embedding_item(item_idx)
        x = torch.cat([user_embedding, item_embedding], dim=-1)
        for fc_layer in self.fc_layers:
            x = fc_layer(x)
            x = nn.ReLU()(x)
        x = self.my_layer(x)
        x = self.fc_output(x)
        return x


class FedMLPModel(FedModelBase):
    def __init__(self,
                 triad,
                 loss_fn,
                 n_user,
                 n_item,
                 dim,
                 layers=[32, 16, 8],
                 output_dim=1,
                 use_gpu=True,
                 optimizer="adam") -> None:
        self.device = ("cuda" if
                       (use_gpu and torch.cuda.is_available()) else "cpu")
        self.name = __class__.__name__
        # self._model = FedMLP(n_user, n_item, dim, layers, output_dim)
        self._model = FedMLP2(n_user, n_item)

        self.server = Server()
        self.clients = Clients(triad, self._model, self.device)

        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.logger = TNLog(self.name)
        self.logger.initial_logger()

    def _check(self, iterator):
        assert abs(sum(iterator) - 1) <= 1e-4

    def update_selected_clients(self, sampled_client_indices, lr, s_params):
        return super().update_selected_clients(sampled_client_indices, lr,
                                               s_params)

    # todo how to add loss?
    def fit(self, epochs, lr, test_triad, fraction=1):
        best_train_loss = None
        is_best = False
        for epoch in tqdm(range(epochs), desc="Training Epochs"):

            # 0. Get params from server
            s_params = self.server.params if epoch != 0 else self._model.state_dict(
            )

            # 1. Select some clients
            sampled_client_indices = self.clients.sample_clients(fraction)

            # 2. Selected clients train
            collector, loss_list, selected_total_size = self.update_selected_clients(
                sampled_client_indices, lr, s_params)

            # 3. Update params to Server
            mixing_coefficients = [
                self.clients[idx].n_item / selected_total_size
                for idx in sampled_client_indices
            ]
            self._check(mixing_coefficients)
            self.server.upgrade_wich_cefficients(collector,
                                                 mixing_coefficients,"asdfsadf")

            self.logger.info(
                f"[{epoch}/{epochs}] Loss:{sum(loss_list)/len(loss_list):>3.5f}"
            )

            print(list(self.clients[0].loss_list))
            if not best_train_loss:
                best_train_loss = sum(loss_list) / len(loss_list)
                is_best = True
            elif sum(loss_list) / len(loss_list) < best_train_loss:
                best_train_loss = sum(loss_list) / len(loss_list)
                is_best = True
            else:
                is_best = False

            ckpt = {
                "model": self.server.params,
                "epoch": epoch + 1,
                "best_loss": best_train_loss
            }
            save_checkpoint(ckpt, is_best, f"output/{self.name}",
                            f"loss_{best_train_loss:.4f}.ckpt")

            if (epoch + 1) % 20 == 0:
                y_list, y_pred_list = self.predict(test_triad)
                mae_ = mae(y_list, y_pred_list)
                mse_ = mse(y_list, y_pred_list)
                rmse_ = rmse(y_list, y_pred_list)

                self.logger.info(
                    f"Epoch:{epoch+1} mae:{mae_},mse:{mse_},rmse:{rmse_}")

    def predict(self, test_loader, resume=False, path=None):
        if resume:
            ckpt = load_checkpoint(path)
            s_params = ckpt["model"]
            self._model.load_state_dict(s_params)
            self.logger.debug(
                f"Check point restored! => loss {ckpt['best_loss']:>3.5f} Epoch {ckpt['epoch']}"
            )
        else:
            s_params = self.server.params
            self._model.load_state_dict(s_params)

        self._model.load_state_dict(s_params)
        y_pred_list = []
        y_list = []
        self._model.to(self.device)
        self._model.eval()
        with torch.no_grad():
            for batch_id, batch in tqdm(enumerate(test_loader),
                                        desc="Model Predict"):
                user, item, rate = batch[0].to(self.device), batch[1].to(
                    self.device), batch[2].to(self.device)
                y_pred = self._model(user, item).squeeze()
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

    def parameters(self):
        return self._model.parameters()

    def __repr__(self) -> str:
        return str(self._model)
