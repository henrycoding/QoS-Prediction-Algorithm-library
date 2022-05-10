import torch
import torch.nn as nn
from torch.nn.init import normal_

from scdmlab.models.abstract_model import GeneralModel
from scdmlab.models.layers import MLPLayers
from scdmlab.utils import InputType


# TODO 应该被归类为 context-aware model
class LDCF(GeneralModel):
    input_type = InputType.INFO

    def __init__(self, config, dataset):
        super(LDCF, self).__init__(config, dataset)

        self.embedding_size = config['embedding_size']
        self.mlp_hidden_size = config['mlp_hidden_size']
        self.dropout_prob = config['dropout_prob']

        self.user_id_embedding = nn.Embedding(self.num_users, self.embedding_size)
        self.user_region_embedding = nn.Embedding(31, self.embedding_size)
        self.service_id_embedding = nn.Embedding(self.num_services, self.embedding_size)
        self.service_region_embedding = nn.Embedding(74, self.embedding_size)

        self.mlp_layers = MLPLayers([4 * self.embedding_size] + self.mlp_hidden_size, self.dropout_prob)
        self.mlp_layers.logger = None
        self.predict_layer = nn.Linear(self.mlp_hidden_size[-1], 1)

        self.loss = nn.L1Loss()

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            normal_(module.weight.data, mean=0.0, std=0.01)

    def forward(self, user, service):
        user_id_embedding = self.user_id_embedding(user[:, 0])
        user_region_embedding = self.user_region_embedding(user[:, 1])
        service_id_embedding = self.service_id_embedding(service[:, 0])
        service_region_embedding = self.service_region_embedding(service[:, 1])

        mlp_input = torch.cat(
            (user_id_embedding, user_region_embedding, service_id_embedding, service_region_embedding), -1)
        mlp_output = self.mlp_layers(mlp_input)
        output = self.predict_layer(mlp_output)

        return output.squeeze(-1)

    def calculate_loss(self, users, services, ratings):
        predicts = self.forward(users, services)
        return self.loss(predicts, ratings)

    def predict(self, users, services):
        return self.forward(users, services)
