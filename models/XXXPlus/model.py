import torch
from data import ToTorchDataset
from models.base.base import ModelBase
from models.base.fedbase import FedModelBase
from torch import nn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from utils.evaluation import mae, mse, rmse
from utils.model_util import load_checkpoint, save_checkpoint, split_d_triad
from utils.mylogger import TNLog

from .client import Clients
from .model_utils import *
from .resnet_utils import *
from .server import Server


class XXXPlus(nn.Module):
    def __init__(self,
                 user_embedding_params,
                 item_embedding_params,
                 in_size,
                 output_size,
                 blocks_size,
                 deepths,
                 activation,
                 personal_layer="my_layer",
                 linear_layers=[128, 64, 32],
                 output_dim=1) -> None:
        super().__init__()

        # embedding
        self.user_embedding = Embedding(**user_embedding_params)
        self.item_embedding = Embedding(**item_embedding_params)

        self.decrease_encoder = ResNetEncoder(in_size=in_size,
                                              blocks_sizes=blocks_size,
                                              deepths=deepths,
                                              activation=activation)
        self.increase_encoder = ResNetEncoder_v2(
            output_size=output_size,
            blocks_sizes=blocks_size[::-1],
            deepths=deepths,
            activation=activation)

        # decoder
        self.fc_layers = nn.Sequential(*[
            Linear(in_size, out_size, activation,personal_layer+f"_{idx}")
            for idx, (in_size, out_size) in enumerate(zip(linear_layers, linear_layers[1:]))
        ])

        # output
        self.output_layers = nn.Linear(linear_layers[-1], output_dim)

    def forward(self, user_idxes: list, item_idxes: list):
        user_embedding = self.user_embedding(user_idxes)
        item_embedding = self.item_embedding(item_idxes)
        x = torch.cat([user_embedding, item_embedding], dim=1)
        x1, y = self.decrease_encoder(x)
        x2 = self.increase_encoder(x1, y)
        x = torch.cat([x1, x2], dim=-1)
        x = self.fc_layers(x)
        x = self.output_layers(x)
        return x


class XXXPlusModel(ModelBase):
    """非联邦的版本
    """
    def __init__(self,
                 user_params,
                 item_params,
                 in_size,
                 output_size,
                 blocks_size,
                 deepths,
                 loss_fn,
                 activation,
                 linear_layers,
                 output_dim=1,
                 use_gpu=True) -> None:
        super().__init__(loss_fn, use_gpu)
        self.model = XXXPlus(user_params, item_params, in_size, output_size,
                             blocks_size, deepths, activation, linear_layers,
                             output_dim)

        self.name = __class__.__name__

    def parameters(self):
        return self.model.parameters()

    def __str__(self) -> str:
        return str(self.model)

    def __repr__(self) -> str:
        return repr(self.model)


# 联邦
class FedXXXLaunch(FedModelBase):
    """联邦的版本
    """
    def __init__(self,
                 user_embedding_params,
                 item_embedding_params,
                 in_size,
                 output_size,
                 blocks_size,
                 batch_size,
                 deepths,
                 activation,
                 d_triad,
                 loss_fn,
                 local_epoch,
                 linear_layers,
                 personal_layer,
                 output_dim=1,
                 optimizer="adam",
                 use_gpu=True) -> None:
        self.device = ("cuda" if
                       (use_gpu and torch.cuda.is_available()) else "cpu")
        self.name = __class__.__name__
        self._model = XXXPlus(user_embedding_params,
                              item_embedding_params,
                              in_size,
                              output_size,
                              blocks_size,
                              deepths,
                              activation,
                              linear_layers,
                              personal_layer=personal_layer,
                              output_dim=output_dim)
        self.server = Server()
        self.clients = Clients(d_triad, self._model, self.device, batch_size,
                               local_epoch)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.personal_layer = personal_layer
        self.logger = TNLog(self.name)
        self.logger.initial_logger()


    def fit(self, epochs, lr, test_d_triad, fraction=1, save_filename=""):
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
                                                 mixing_coefficients)
            # self.server.upgrade_average(collector)

            # 3. 服务端根据参数更新模型
            self.logger.info(
                f"[{epoch}/{epochs}] Loss:{sum(loss_list)/len(loss_list):>3.5f}"
            )

            print(self.clients[0].loss_list)
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

            if (epoch + 1) % 10 == 0:
                y_list, y_pred_list = self.predict(test_d_triad)
                mae_ = mae(y_list, y_pred_list)
                mse_ = mse(y_list, y_pred_list)
                rmse_ = rmse(y_list, y_pred_list)

                self.logger.info(
                    f"Epoch:{epoch+1} mae:{mae_},mse:{mse_},rmse:{rmse_}")

    # 这里的代码写的很随意 没时间优化了
    def predict(self, d_triad, resume=False, path=None):
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
        y_pred_list = []
        y_list = []
        triad, p_triad = split_d_triad(d_triad)
        p_triad_dataloader = DataLoader(ToTorchDataset(p_triad),
                                        batch_size=2048)  # 这里可以优化 这样写不是很好

        self._model.to(self.device)
        self._model.eval()
        with torch.no_grad():
            # for batch_id, batch in tqdm(enumerate(test_loader)):
            for batch_id, batch in tqdm(enumerate(p_triad_dataloader),ncols=80,
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

    def __str__(self) -> str:
        return str(self._model)

    def __repr__(self) -> str:
        return repr(self._model)
