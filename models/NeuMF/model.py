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
        self.logistic = nn.Sigmoid()

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
        logits = self.affine_output(vector)
        rating = self.logistic(logits)

        return rating

    def load_pretrain_weights(self):
        density = self.config.TRAIN.DENSITY

        gmf_model_dir = absolute(self.config.TRAIN.GMF_MODEL_DIR.format(density=density))
        mlp_model_dir = absolute(self.config.TRAIN.MLP_MODEL_DIR.format(density=density))

        # FIXME 加载预训练模型是否需要放到GPU上
        # device = get_device(config)
        # gmf_model.to(device)

        gmf_ckpt = load_checkpoint(gmf_model_dir)
        gmf_model_config = get_gmf_config()
        gmf_model_config.defrost()
        gmf_model_config.TRAIN.NUM_USERS = self.config.TRAIN.NUM_USERS
        gmf_model_config.TRAIN.NUM_ITEMS = self.config.TRAIN.NUM_ITEMS
        gmf_model_config.TRAIN.DENSITY = self.config.TRAIN.DENSITY
        gmf_model_config.freeze()
        gmf_model = GMF(gmf_model_config)
        gmf_model.load_state_dict(gmf_ckpt['model'])

        self.embedding_user_gmf.weight.data = gmf_model.embedding_user.weight.data
        self.embedding_item_gmf.weight.data = gmf_model.embedding_item.weight.data

        mlp_ckpt = load_checkpoint(mlp_model_dir)
        mlp_model_config = get_mlp_config()
        mlp_model_config.defrost()
        mlp_model_config.TRAIN.NUM_USERS = self.config.TRAIN.NUM_USERS
        mlp_model_config.TRAIN.NUM_ITEMS = self.config.TRAIN.NUM_ITEMS
        mlp_model_config.TRAIN.DENSITY = self.config.TRAIN.DENSITY
        mlp_model_config.freeze()
        mlp_model = MLP(mlp_model_config)
        mlp_model.load_state_dict(mlp_ckpt['model'])

        self.embedding_user_mlp.weight.data = mlp_model.embedding_user.weight.data
        self.embedding_item_mlp.weight.data = mlp_model.embedding_item.weight.data
        for idx in range(len(self.fc_layers)):
            self.fc_layers[idx].weight.data = mlp_model.fc_layers[idx].weight.data

        self.affine_output.weight.data = 0.5 * torch.cat(
            [gmf_model.affine_output.weight.data, mlp_model.affine_output.weight.data], dim=-1)
        self.affine_output.bias.data = 0.5 * (gmf_model.affine_output.bias.data + mlp_model.affine_output.bias.data)


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
    cfg = get_cfg_defaults()
    test = ModelTest(NeuMFModel, cfg)
    test.run()
