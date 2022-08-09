import torch
import torch.nn.functional as F
from torch import nn
from yacs.config import CfgNode
from tqdm import tqdm
from models.LDCF.client import Clients
from models.LDCF.server import Server
from models.base import FedModelBase, ModelBase
from utils import TNLog
from utils.evaluation import mae, rmse, mse
from utils.model_util import save_checkpoint, load_checkpoint, use_loss_fn


class LDCF(nn.Module):
    def __init__(self, config: CfgNode) -> None:
        super(LDCF, self).__init__()
        self.config = config

        self.num_users = self.config.TRAIN.NUM_USERS
        self.num_items = self.config.TRAIN.NUM_ITEMS
        self.num_users_ac = self.config.TRAIN.NUM_USERS_AC
        self.num_item_ac = self.config.TRAIN.NUM_ITEM_AC

        self.layers = self.config.TRAIN.LAYERS
        self.latent_dim = self.config.TRAIN.LATENT_DIM
        # MLP
        self.embedding_user_mlp = nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item_mlp = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)

        # AC
        self.embedding_user_ac = nn.Embedding(num_embeddings=self.num_users_ac, embedding_dim=self.latent_dim)
        self.embedding_item_ac = nn.Embedding(num_embeddings=self.num_item_ac, embedding_dim=self.latent_dim)

        # fully-connected layers in MLP
        self.fc_layers = nn.ModuleList()
        for in_size, out_size in zip(self.layers[:-1], self.layers[1:]):
            self.fc_layers.append(nn.Linear(in_size, out_size))
        self.affine_output = nn.Linear(2 + self.layers[-1], 1)
        self.logistic = nn.Sigmoid()

    def forward(self, user, item):
        user_id = user[:, 0]
        user_lc = user[:, 1:3]
        item_id = item[:, 0]
        item_lc = item[:, 1:3]
        user_embedding_mlp = self.embedding_user_mlp(user_id)
        item_embedding_mlp = self.embedding_item_mlp(item_id)

        user_embedding_ac = self.embedding_user_ac(user_lc)
        item_embedding_ac = self.embedding_item_ac(item_lc)

        sim_vec = F.cosine_similarity(user_embedding_ac, item_embedding_ac, dim=-1)

        mlp_vec = torch.cat([user_embedding_mlp, item_embedding_mlp], dim=-1)  # 拼接

        for layer in self.fc_layers:
            mlp_vec = layer(mlp_vec)
            mlp_vec = F.relu(mlp_vec)

        vector = torch.cat([sim_vec, mlp_vec], dim=-1)
        logits = self.affine_output(vector)
        # rating = self.logistic(logits)
        return logits

class LDCFModel(ModelBase):
    def __init__(self, config: CfgNode, writer=None) -> None:
        model = LDCF(config)
        super(LDCFModel, self).__init__(model, config, writer)

    def parameters(self):
        return self.model.parameters()

    def __repr__(self) -> str:
        return str(self.model)

'''
          0.05      0.10      0.15      0.20
MAE   0.507551  0.461609  0.411447  0.380726
MSE   2.040553  1.968015  1.738773  1.616560
RMSE  1.428479  1.402860  1.318625  1.271440
'''



class FedLDCFModel(FedModelBase):
    def __init__(self, triad, cfg: CfgNode) -> None:
        super().__init__(cfg)
        use_gpu = self.config.SYSTEM.USE_GPU
        self.device = ("cuda" if
                       (use_gpu and torch.cuda.is_available()) else "cpu")
        self.name = __class__.__name__
        self._model = LDCF(cfg)
        self.server = Server(cfg)
        self.clients = Clients(cfg, triad, self._model, self.device)
        self.optimizer = self.config.TRAIN.OPTIMIZER.TYPE
        self.loss_fn = use_loss_fn(self.config)
        self.logger = TNLog(self.name)
        self.logger.initial_logger()

    def _check(self, iterator):
        assert abs(sum(iterator) - 1) <= 1e-4  # 验证加权系数和

    def fit(self, epochs, lr, test_triad, date, density):
        fraction = self.config.TRAIN.FRACTION
        best_train_loss = None
        is_best = False
        for epoch in tqdm(range(epochs), desc=f"Density={density},Training Epochs"):
            # 0. Get params from server
            s_params = self.server.params if epoch != 0 else self._model.state_dict(
            )

            # 1. Select some clients
            sampled_client_indices = self.clients.sample_clients(fraction)

            # 2. Selected clients train
            collector, loss_list, selected_total_size = self.update_selected_clients(
                sampled_client_indices, lr, s_params)

            # 3. Update params to Server
            # 服务端加权平均更新系数
            mixing_coefficients = [
                self.clients[idx].n_item / selected_total_size
                for idx in sampled_client_indices
            ]
            self._check(mixing_coefficients)
            self.server.upgrade_wich_cefficients(collector,
                                                 mixing_coefficients)

            self.logger.info(
                f"[{epoch}/{epochs}] Loss:{sum(loss_list) / len(loss_list):>3.5f}"
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

            save_checkpoint(ckpt, is_best, f"output/{self.name}/{date}/density-{density}",
                            f"loss_{best_train_loss:.4f}.ckpt")

            if (epoch + 1) % 25 == 0:
                y_list, y_pred_list = self.predict(test_triad)
                mae_ = mae(y_list, y_pred_list)
                mse_ = mse(y_list, y_pred_list)
                rmse_ = rmse(y_list, y_pred_list)

                self.logger.info(
                    f"Epoch:{epoch + 1} mae:{mae_},mse:{mse_},rmse:{rmse_}")

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
