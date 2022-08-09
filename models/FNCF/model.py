import torch
from torch import nn
from yacs.config import CfgNode
import torch.nn.functional as F

from models.base import ModelBase
from models.FNCF.config import get_cfg_defaults
from utils.model_util import ModelTest


class NeuMF(nn.Module):
    def __init__(self, config: CfgNode) -> None:
        super(NeuMF, self).__init__()
        self.config = config

        self.num_users = self.config.TRAIN.NUM_USERS
        self.num_items = self.config.TRAIN.NUM_ITEMS
        self.layers = self.config.TRAIN.LAYERS
        self.latent_dim_gmf = self.config.TRAIN.LATENT_DIM_GMF
        self.latent_dim_mlp = self.config.TRAIN.LATENT_DIM_MLP
        # GMF
        self.embedding_user_gmf = nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim_gmf)
        self.embedding_item_gmf = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim_gmf)

        # MLP
        self.embedding_user_mlp = nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim_mlp)
        self.embedding_item_mlp = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim_mlp)
        # fully-connected layers in MLP
        self.fc_layers = nn.ModuleList()
        for in_size, out_size in zip(self.layers[:-1], self.layers[1:]):
            self.fc_layers.append(nn.Linear(in_size, out_size))
        self.affine_output = nn.Linear(self.latent_dim_gmf + self.layers[-1], 1)
        self.logistic = nn.Sigmoid()

    def forward(self, user_indices, item_indices):
        user_embedding_gmf = self.embedding_user_gmf(user_indices)
        item_embedding_gmf = self.embedding_item_gmf(item_indices)
        user_embedding_mlp = self.embedding_user_mlp(user_indices)
        item_embedding_mlp = self.embedding_item_mlp(item_indices)

        gmf_vec = torch.mul(user_embedding_gmf, item_embedding_gmf)  # 点乘
        mlp_vec = torch.cat([user_embedding_mlp, item_embedding_mlp], dim=-1)  # 拼接

        for layer in self.fc_layers:
            mlp_vec = layer(mlp_vec)
            mlp_vec = F.relu(mlp_vec)

        vector = torch.cat([gmf_vec, mlp_vec], dim=-1)
        logits = self.affine_output(vector)
        # rating = self.logistic(logits)

        return logits


class NeuMFModel(ModelBase):
    def __init__(self, config: CfgNode, writer=None):
        model = NeuMF(config)
        super().__init__(model, config, writer)


def set_train_param(parameters):
    cfg = get_cfg_defaults()
    cfg.defrost()
    cfg.TRAIN.DATA_TYPE = parameters['model']['dataset']
    density_list = [parameters['model']['density']]
    cfg.TRAIN.DENSITY_LIST = density_list
    cfg.TRAIN.BATCH_SIZE = parameters['model']['batchSize']
    cfg.TRAIN.LATENT_DIM_GMF = parameters['model']['latentDim']
    cfg.TRAIN.LATENT_DIM_MLP = parameters['model']['latentDim']
    cfg.TRAIN.NUM_EPOCHS = parameters['model']['epoch']
    cfg.TRAIN.LOSS_FN.TYPE = parameters['model']['lossFn']
    cfg.TRAIN.OPTIMIZER.TYPE = parameters['model']['opt']
    cfg.freeze()
    return cfg


def start_train_NeuMF(parameters):
    cfg = set_train_param(parameters)
    test = ModelTest(NeuMFModel, cfg)
    test.run()


if __name__ == '__main__':
    cfg = get_cfg_defaults()
    test = ModelTest(NeuMFModel, cfg)
    test.run()

'''
          0.05      0.10      0.15      0.20
MAE   0.442713  0.361492  0.334233  0.331722
MSE   1.877207  1.641176  1.608249  1.582819
RMSE  1.370112  1.281084  1.268167  1.258101
'''