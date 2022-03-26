import torch
from models.base import ModelBase
from torch import nn


class GMF(nn.Module):
    def __init__(self,
                 n_user,
                 n_item,
                 dim=8,
                 fc_layers=[64, 32, 16],
                 output_dim=1) -> None:
        super(GMF, self).__init__()

        self.num_users = n_user
        self.num_items = n_item
        self.latent_dim = dim
        self.layers = fc_layers

        self.embedding_user = nn.Embedding(num_embeddings=self.num_users,
                                           embedding_dim=self.latent_dim)
        self.embedding_item = nn.Embedding(num_embeddings=self.num_items,
                                           embedding_dim=self.latent_dim)

        self.fc_layers = nn.ModuleList()

        for idx, (in_size, out_size) in enumerate(
                zip(self.layers[:-1], self.layers[1:])):
            self.fc_layers.append(nn.Linear(in_size, out_size))

        self.fc_output = nn.Linear(self.layers[-1], output_dim)

    def forward(self, user_idx, item_idx):
        user_embedding = self.embedding_user(user_idx)
        item_embedding = self.embedding_item(item_idx)
        x = torch.mul(user_embedding, item_embedding)
        for fc_layer in self.fc_layers:
            x = fc_layer(x)
            x = nn.GELU()(x)
            x = nn.Dropout()(x)

        x = self.fc_output(x)
        return x


class GMFModel(ModelBase):
    def __init__(self,
                 loss_fn,
                 n_user,
                 n_item,
                 dim=8,
                 output_dim=1,
                 use_gpu=True,
                 layers=[64, 32]):
        super().__init__(loss_fn, use_gpu)
        self.model = GMF(n_user, n_item, dim, layers, output_dim)
        if use_gpu:
            self.model.to(self.device)

    def parameters(self):
        return self.model.parameters()

    def __repr__(self) -> str:
        return str(self.model)
