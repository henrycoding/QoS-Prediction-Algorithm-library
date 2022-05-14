import numpy as np
import torch
import torch.nn as nn
from torch.nn.init import normal_

from scdmlab.models.abstract_model import GeneralModel
from scdmlab.models.layers import MLPLayers, ResidualLayer
from scdmlab.utils import InputType


def res_net(input_size, hidden_size):
    b1 = ResidualLayer(input_size, hidden_size[0])
    blks = []
    for i in range(len(hidden_size) - 1):
        blks.append(ResidualLayer(hidden_size[i], hidden_size[i + 1]))
    return nn.Sequential(b1, *blks)


class resnet(GeneralModel):
    input_type = InputType.MATRIX

    def __init__(self, config, dataset):
        super(resnet, self).__init__(config, dataset)
        self.embedding_size = config['embedding_size']
        self.hidden_size = config['hidden_size']

        self.user_embedding = nn.Embedding(self.num_users, self.embedding_size)
        self.service_embedding = nn.Embedding(self.num_services, self.embedding_size)

        self.user_latent_vector = np.empty((self.num_users, self.hidden_size[-1]))
        self.service_latent_vector = np.empty((self.num_services, self.hidden_size[-1]))

        self.user_layers = res_net(self.embedding_size, self.hidden_size)
        self.serivce_layers = res_net(self.embedding_size, self.hidden_size)
        self.predict_layer = nn.Linear(self.hidden_size[-1] * 2, 1)

        # define loss
        self.loss = nn.L1Loss()

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            normal_(module.weight.data, mean=0.0, std=0.01)

    def _update_latent_vector(self, user, service, u, s):
        for i in range(len(user)):
            self.user_latent_vector[user[i]] = u[i]
            self.service_latent_vector[service[i]] = s[i]

    def forward(self, user, service):
        user_embedding = self.user_embedding(user)
        service_embedding = self.service_embedding(service)

        u = nn.Sequential(*self.user_layers)(user_embedding)
        s = nn.Sequential(*self.serivce_layers)(service_embedding)

        self._update_latent_vector(user.detach().cpu().numpy(), service.detach().cpu().numpy(),
                                   u.detach().cpu().numpy(), s.detach().cpu().numpy())

        output_vector = torch.cat([u, s], dim=1)
        output = self.predict_layer(output_vector)
        return output

    def calculate_loss(self, user, service, rating):
        predict = self.forward(user, service)
        return self.loss(predict, rating)

    def predict(self, user, service):
        with torch.no_grad():
            user_latent_vector = torch.tensor(self.user_latent_vector)
            service_latent_vector = torch.tensor(self.service_latent_vector)
            u = user_latent_vector[user].float()
            s = service_latent_vector[service].float()
            u = u.to(self.device)
            s = s.to(self.device)
            predict = self.predict_layer(torch.cat([u, s], dim=1))
            return predict
        # return self.forward(user, service)
