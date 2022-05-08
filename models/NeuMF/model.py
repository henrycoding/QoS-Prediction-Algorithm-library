import argparse
import os
import sys

project_path = 'D:\SHUlpt\Code\QoS-Predcition-Algorithm-library'
os.chdir(project_path)
sys.path.append(project_path)

import torch
from torch import nn
import torch.nn.functional as F
from yacs.config import CfgNode

from models.base import ModelBase

# config
from models.NeuMF.config import get_cfg_defaults
from models.GMF.config import get_cfg_defaults as get_gmf_config
from models.MLP.config import get_cfg_defaults as get_mlp_config

# model utils
from utils.model_util import ModelTest, load_checkpoint

from root import absolute

from models.GMF.model import GMF, GMFModel
from models.MLP.model import MLP

# send the request to the back end
from utils.request import send_pid


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

        # fully-connected layers
        self.fc_layers = nn.ModuleList()
        for in_size, out_size in zip(self.layers[:-1], self.layers[1:]):
            self.fc_layers.append(nn.Linear(in_size, out_size))

        self.affine_output = nn.Linear(self.latent_dim_gmf + self.layers[-1], 1)

    def forward(self, user_indices, item_indices):
        user_embedding_gmf = self.embedding_user_gmf(user_indices)
        item_embedding_gmf = self.embedding_item_gmf(item_indices)
        user_embedding_mlp = self.embedding_user_mlp(user_indices)
        item_embedding_mlp = self.embedding_item_mlp(item_indices)

        gmf_vec = torch.mul(user_embedding_gmf, item_embedding_gmf)
        mlp_vec = torch.cat([user_embedding_mlp, item_embedding_mlp], dim=-1)

        for layer in self.fc_layers:
            mlp_vec = layer(mlp_vec)
            mlp_vec = F.relu(mlp_vec)

        vector = torch.cat([gmf_vec, mlp_vec], dim=-1)
        rating = self.affine_output(vector)

        return rating


class NeuMFModel(ModelBase):
    def __init__(self, config: CfgNode, writer=None) -> None:
        model = NeuMF(config)

        try:
            resume = config.TRAIN.PRETRAIN
        except:
            resume = False
        if resume:
            model.load_pretrain_weights()

        super(NeuMFModel, self).__init__(model, config, writer)

    def parameters(self):
        return self.model.parameters()

    def __repr__(self) -> str:
        return str(self.model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=int)
    args, _ = parser.parse_known_args()

    cfg = get_cfg_defaults()
    cfg.defrost()
    cfg.SYSTEM.ID = args.id
    cfg.freeze()

    # send pid to the back end
    send_pid(cfg.SYSTEM.ID, os.getpid())

    test = ModelTest(NeuMFModel, cfg)
    test.run()
