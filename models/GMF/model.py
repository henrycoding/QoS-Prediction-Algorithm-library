import torch
from yacs.config import CfgNode

from models.GMF.config import get_cfg_defaults
from models.base import ModelBase
from torch import nn

from utils.model_util import use_cuda, ModelTest


class GMF(nn.Module):
    def __init__(self, config):
        super(GMF, self).__init__()
        self.config = config

        self.num_users = self.config.TRAIN.NUM_USERS
        self.num_items = self.config.TRAIN.NUM_ITEMS
        self.latent_dim = self.config.TRAIN.LATENT_DIM

        self.embedding_user = nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)

        # 逻辑回归层
        self.affine_output = nn.Linear(self.latent_dim, 1)
        self.logistic = nn.Sigmoid()

    def forward(self, user_indices, item_indices):
        # 用户和物品的embedding
        user_embedding = self.embedding_user(user_indices)
        item_embedding = self.embedding_item(item_indices)

        # 隐向量做点积
        element_product = torch.mul(user_embedding, item_embedding)

        # 逻辑回归
        logits = self.affine_output(element_product)
        rating = self.logistic(logits)
        return rating


class GMFModel(ModelBase):
    def __init__(self, config: CfgNode, writer=None) -> None:
        model = GMF(config)
        super(GMFModel, self).__init__(model, config, writer)

    def parameters(self):
        return self.model.parameters()

    def __repr__(self) -> str:
        return str(self.model)


if __name__ == "__main__":
    cfg = get_cfg_defaults()
    test = ModelTest(GMFModel, cfg)
    test.run()
