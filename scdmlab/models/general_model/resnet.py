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
    input_type = InputType.INFO

    def __init__(self, config, dataset):
        super(resnet, self).__init__(config, dataset)
        self.embedding_size = config['embedding_size']
        self.hidden_size = config['hidden_size']

        self.user_id_embedding = nn.Embedding(self.num_users, self.embedding_size)
        self.user_country_embedding = nn.Embedding(31, self.embedding_size)
        self.user_as_embedding = nn.Embedding(137, self.embedding_size)

        self.service_id_embedding = nn.Embedding(self.num_services, self.embedding_size)
        self.service_country_embedding = nn.Embedding(74, self.embedding_size)
        self.service_as_embedding = nn.Embedding(992, self.embedding_size)

        self.user_latent_vector = np.empty((self.num_users, self.hidden_size[-1]))
        self.service_latent_vector = np.empty((self.num_services, self.hidden_size[-1]))

        user_input_size = 3 * self.embedding_size + 2
        service_input_size = 3 * self.embedding_size + 2

        self.user_latent_vector = np.empty((339, user_input_size))
        self.service_latent_vector = np.empty((5825, service_input_size))
        self.user_layers = res_net(user_input_size, self.hidden_size)
        self.service_layers = res_net(service_input_size, self.hidden_size)

        self.predict_layer = nn.Linear(user_input_size + service_input_size, 1)

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

        self.user_latent_vectors = torch.tensor(self.user_latent_vector).to(self.device)
        self.service_latent_vector = torch.tensor(self.service_latent_vector).to(self.device)

    def forward(self, user, service):
        user_id_embedding = self.user_id_embedding(user[:, 0])
        user_country_embedding = self.user_country_embedding(user[:, 1])
        user_as_embedding = self.user_as_embedding(user[:, 2])
        user_loc = user[:, [3, 4]]

        service_id_embedding = self.service_id_embedding(service[:, 0])
        service_country_embedding = self.service_country_embedding(service[:, 1])
        service_as_embedding = self.service_as_embedding(service[:, 2])
        service_loc = service[:, [3, 4]]

        user_latent_vector = torch.cat([user_id_embedding, user_country_embedding, user_as_embedding, user_loc], dim=1)
        service_latent_vector = torch.cat([service_id_embedding, service_country_embedding,service_as_embedding, service_loc], dim=1)


        u = nn.Sequential(*self.user_layers)(user_latent_vector)
        s = nn.Sequential(*self.serivce_layers)(service_latent_vector)

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
            uid = user[:, 0].float()
            sid = service[:, 0].float()
            u = user_latent_vector[uid].float()
            s = service_latent_vector[sid].float()

            u = u.to(self.device)
            s = s.to(self.device)
            predict = self.predict_layer(torch.cat([u, s], dim=1))
            return predict
