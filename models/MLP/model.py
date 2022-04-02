import torch
from yacs.config import CfgNode
from models.base import ModelBase
from torch import nn
import torch.nn.functional as F
from models.MLP.config import get_cfg_defaults
from utils.model_util import ModelTest, load_checkpoint, get_device
from models.GMF.model import GMFModel, GMF
from root import absolute


class MLP(nn.Module):
    def __init__(self, config) -> None:
        super(MLP, self).__init__()
        self.config = config
        self.layers = self.config.TRAIN.LAYERS

        self.num_users = self.config.TRAIN.NUM_USERS
        self.num_items = self.config.TRAIN.NUM_ITEMS
        self.latent_dim = self.config.TRAIN.LATENT_DIM

        self.embedding_user = nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)

        # fully-connected layers
        self.fc_layers = nn.ModuleList()
        for in_size, out_size in zip(self.layers[:-1], self.layers[1:]):
            self.fc_layers.append(nn.Linear(in_size, out_size))

        self.affine_output = nn.Linear(self.layers[-1], 1)
        self.logistic = nn.Sigmoid()

    def forward(self, user_idx, item_idx):
        user_embedding = self.embedding_user(user_idx)
        item_embedding = self.embedding_item(item_idx)

        vector = torch.cat([user_embedding, item_embedding], dim=-1)  # the concat latent vector
        for layer in self.fc_layers:
            vector = layer(vector)
            vector = F.relu(vector)
            # TODO BatchNorm层怎么加
            # vector = nn.BatchNorm1d()(vector)
            # vector = nn.Dropout(p=0.5)(vector)
        logits = self.affine_output(vector)
        rating = self.logistic(logits)
        return rating

    def load_pretrain_weights(self):
        try:
            density = self.config.TRAIN.DENSITY
        except Exception:
            raise Exception("THe parameter 'TRAIN.DENSITY' is not found!")

        try:
            model_dir = absolute(self.config.TRAIN.PRETRAIN_DIR.format(density=density))
        except Exception:
            raise Exception("The 'TRAIN.PRETRAIN_DIR' is not provided in the configuration file!")

        ckpt = load_checkpoint(model_dir)
        gmf_model = GMF(self.config)
        gmf_model.load_state_dict(ckpt['model'])

        # FIXME 加载预训练模型是否需要放到GPU上
        # device = get_device(config)
        # gmf_model.to(device)

        self.embedding_user.weight.data = gmf_model.embedding_user.weight.data
        self.embedding_item.weight.data = gmf_model.embedding_item.weight.data


class MLPModel(ModelBase):
    def __init__(self, config: CfgNode, writer=None) -> None:
        model = MLP(config)

        try:
            resume = config.TRAIN.PRETRAIN
        except:
            resume = False
        if resume:
            model.load_pretrain_weights(config)

        super(MLPModel, self).__init__(model, config, writer)

    def parameters(self):
        return self.model.parameters()

    def __repr__(self) -> str:
        return str(self.model)


if __name__ == "__main__":
    cfg = get_cfg_defaults()
    test = ModelTest(MLPModel, cfg)
    test.run()
