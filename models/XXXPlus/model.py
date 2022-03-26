import torch
from models.base.base import ModelBase
from torch import nn

from .model_utils import *
from .resnet_utils import *


class XXXPlus(nn.Module):
    def __init__(self,
                 user_embedding_params,
                 item_embedding_params,
                 in_size,
                 output_size,
                 blocks_size,
                 deepths,
                 output_dim=1) -> None:
        super().__init__()

        # embedding
        self.user_embedding = Embedding(**user_embedding_params)
        self.item_embedding = Embedding(**item_embedding_params)

        self.decrease_encoder = ResNetEncoder(in_size=in_size,
                                              blocks_sizes=blocks_size,
                                              deepths=deepths)
        self.increase_encoder = ResNetEncoder_v2(
            output_size=output_size,
            blocks_sizes=blocks_size[::-1],
            deepths=deepths)

        # decoder
        # self.fc_layers = nn.Sequential(*[
        #     Linear(in_size, out_size, activation)
        #     for in_size, out_size in zip(linear_layers, linear_layers[1:])
        # ])

        # output
        self.output_layers = nn.Linear(output_size, output_dim)

    def forward(self, user_idxes: list, item_idxes: list):
        user_embedding = self.user_embedding(user_idxes)
        item_embedding = self.item_embedding(item_idxes)
        x = torch.cat([user_embedding, item_embedding], dim=1)
        x,y = self.decrease_encoder(x)
        x = self.increase_encoder(x,y)
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
                 output_dim=1,
                 use_gpu=True) -> None:
        super().__init__(loss_fn, use_gpu)
        self.model = XXXPlus(user_params, item_params, in_size, output_size,
                             blocks_size, deepths, output_dim)

        self.name = __class__.__name__

    def parameters(self):
        return self.model.parameters()

    def __str__(self) -> str:
        return str(self.model)

    def __repr__(self) -> str:
        return repr(self.model)
