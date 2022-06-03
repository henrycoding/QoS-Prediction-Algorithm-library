import torch
from models.base import ModelBase
from models.XXXPlus.model_utils import Embedding
from numpy import ulonglong
from torch import nn
from torch.nn.init import normal_


class LDCF(nn.Module):
    def __init__(self,
                 user_embedding_params,
                 item_embedding_params,
                 fc_layers=[64, 32, 16],
                 output_dim=1) -> None:
        super(LDCF, self).__init__()

        self.layers = fc_layers
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
        # self.apply(self._init_weights)

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
        cosine_vector = torch.cosine_similarity(user_lc_latent,item_lc_latent).reshape((-1,1))
        for fc_layer in self.fc_layers:
            x = fc_layer(x)
            x = nn.ReLU()(x)
        x = torch.cat([x, cosine_vector],dim=1)
        x = self.fc_output(x)
        return x


class LDCFModel(ModelBase):
    def __init__(self,
                 loss_fn,
                 user_embedding_params,
                 item_embedding_params,
                 fc_layers,
                 use_gpu=True):
        super().__init__(loss_fn, use_gpu)
        self.model = LDCF(user_embedding_params, item_embedding_params,
                          fc_layers)
        if use_gpu:
            self.model.to(self.device)

    def parameters(self):
        return self.model.parameters()

    def __repr__(self) -> str:
        return str(self.model)
