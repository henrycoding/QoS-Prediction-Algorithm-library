import os
import time

import numpy as np
import torch
from models.base import FedModelBase, ModelBase
from models.FedLDCF.model_utils import Embedding
from numpy.lib.function_base import select
from pandas.io.parsers import read_table
from root import absolute
from torch import nn
from torch.nn.init import normal_
from torch.optim.adam import Adam
from torch.optim.sgd import SGD
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils.evaluation import mae, mse, rmse
from utils.model_util import load_checkpoint, save_checkpoint
from utils.mylogger import TNLog

from .client import Clients
from .server import Server


class FedLDCF(nn.Module):
    def __init__(self,
                 user_embedding_params,
                 item_embedding_params,
                 fc_layers=[64, 32, 16],
                 output_dim=1) -> None:
        super(FedLDCF, self).__init__()

        self.layers = fc_layers
        self.personal_layer = "xxxxxxxxxx"
        self.user_id_dims = user_embedding_params["embedding_dims"][0]
        self.item_id_dims = item_embedding_params["embedding_dims"][0]
        self.user_lc_dims = sum(user_embedding_params["embedding_dims"][1:])
        self.item_lc_dims = sum(item_embedding_params["embedding_dims"][1:])

        # embedding
        # id cy as
        self.user_embedding = Embedding(**user_embedding_params)
        self.item_embedding = Embedding(**item_embedding_params)

        self.fc_layers = nn.ModuleList()

        for idx, (in_size,
                  out_size) in enumerate(zip(self.layers[:-1],
                                             self.layers[1:])):
            self.fc_layers.append(nn.Linear(in_size, out_size))

        self.fc_output = nn.Linear(self.layers[-1] + 1, output_dim)

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            normal_(module.weight.data, mean=0.0, std=0.01)

    def forward(self, user_idx, item_idx):
        user_embedding = self.user_embedding(user_idx)
        item_embedding = self.item_embedding(item_idx)
        x = torch.cat([user_embedding, item_embedding], dim=1)
        # AC-COS
        user_lc_latent = user_embedding[:, self.user_id_dims + 1:]
        item_lc_latent = user_embedding[:, self.item_id_dims + 1:]
        cosine_vector = torch.cosine_similarity(user_lc_latent,
                                                item_lc_latent).reshape(
                                                    (-1, 1))
        for fc_layer in self.fc_layers:
            x = fc_layer(x)
            x = nn.ReLU()(x)
        x = torch.cat([x, cosine_vector], dim=1)
        x = self.fc_output(x)
        return x


# 联邦
class FedLDCFModel(FedModelBase):
    """联邦的版本
    """
    def __init__(self,
                 user_embedding_params,
                 item_embedding_params,
                 loss_fn,
                 d_triad,
                 test_d_triad,
                 optimizer,
                 use_gpu=True) -> None:
        self.device = ("cuda" if
                       (use_gpu and torch.cuda.is_available()) else "cpu")
        self.name = __class__.__name__
        self._model = FedLDCF(user_embedding_params, item_embedding_params)
        self.server = Server()
        self.clients = Clients(d_triad, test_d_triad, self._model, self.device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.logger = TNLog(self.name)
        self.logger.initial_logger()
        self.date = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())

        # Tensorboard
        # 自动打开tensorboard，只要浏览器中打开localhost:6006即可看到训练过程
        save_dir = absolute(f"output/{self.name}/{self.date}/TensorBoard")
        os.makedirs(save_dir)
        self.writer = SummaryWriter(log_dir=save_dir)
        # tensorboard = program.TensorBoard()
        # tensorboard.configure(argv=[None, '--logdir', save_dir])
        # tensorboard.launch()

    def fit(self, epochs, lr, verbose_epoch=10, fraction=1, save_filename=""):
        best_train_loss = None
        is_best = False
        for epoch in tqdm(range(epochs),
                          desc="Traing Epochs ",
                          ncols=80,
                          colour="green"):

            # 0. Get params from server
            s_params = self.server.params if epoch != 0 else self._model.state_dict(
            )
            # 1. Select some clients
            sampled_client_indices = self.clients.sample_clients(fraction)

            # 2. Selected clients train
            collector, loss_list, selected_total_size = self.update_selected_clients(
                sampled_client_indices, lr, s_params)

            # 3. update params to Server
            mixing_coefficients = [
                self.clients[idx].n_item / selected_total_size
                for idx in sampled_client_indices
            ]
            self._check(mixing_coefficients)
            self.server.upgrade_wich_cefficients(collector,
                                                 mixing_coefficients,
                                                 self._model.personal_layer)
            # self.server.upgrade_average(collector)

            # 3. 服务端根据参数更新模型
            self.logger.info(
                f"[{save_filename}] [{epoch}/{epochs}] Loss:{sum(loss_list)/len(loss_list):>3.5f}"
            )

            self.writer.add_scalar("Training Loss",
                                   sum(loss_list) / len(loss_list), epoch + 1)

            # print(self.clients[0].loss_list)
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
            save_checkpoint(
                ckpt, is_best, f"output/{self.name}",
                f"loss_{save_filename}_{best_train_loss:.4f}.ckpt")

            if (epoch + 1) % verbose_epoch == 0:
                client_indices = self.clients.sample_clients(1)
                y_list, y_pred_list = self.evaluation_selected_clients(
                    client_indices, self.server.params)
                mae_ = mae(y_list, y_pred_list)
                mse_ = mse(y_list, y_pred_list)
                rmse_ = rmse(y_list, y_pred_list)

                self.logger.info(
                    f"[{save_filename}] Epoch:{epoch+1} mae:{mae_},mse:{mse_},rmse:{rmse_}"
                )

                self.writer.add_scalar("Test mae", mae_, epoch + 1)
                self.writer.add_scalar("Test rmse", rmse_, epoch + 1)



    def parameters(self):
        return self._model.parameters()

    def __str__(self) -> str:
        return str(self._model)

    def __repr__(self) -> str:
        return repr(self._model)
