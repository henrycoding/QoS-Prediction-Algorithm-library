# import torch
# from torch import nn
# import torch.nn.functional as F
# from yacs.config import CfgNode
#
# from models.base import ModelBase
#
# # config
# from models.NeuMF.config import get_cfg_defaults
# from scdmlab.model.GMF import get_cfg_defaults as get_gmf_config
# from models.MLP.config import get_cfg_defaults as get_mlp_config
#
# # model utils
# from scdmlab.utils import ModelTest, load_checkpoint
#
# from root import absolute
#
# from scdmlab.model.GMF.model import GMF
# from models.MLP.model import MLP
#
#
import torch
import torch.nn as nn
from torch.nn.init import normal_

from scdmlab.models.abstract_model import GeneralModel
from scdmlab.models.layers import MLPLayers
from scdmlab.utils import InputType


class NeuMF(GeneralModel):
    input_type = InputType.MATRIX

    def __init__(self, config, dataset):
        super(NeuMF, self).__init__(config, dataset)

        # load parameters info
        self.gmf_train = config['gmf_train']
        self.mlp_train = config['mlp_train']
        self.gmf_embedding_size = config['gmf_embedding_size']
        self.mlp_embedding_size = config['mlp_embedding_size']
        self.mlp_hidden_size = config['mlp_hidden_size']
        self.dropout_prob = config['dropout_prob']

        # define layers
        self.user_gmf_embedding = nn.Embedding(self.num_users, self.gmf_embedding_size)
        self.item_gmf_embedding = nn.Embedding(self.num_services, self.gmf_embedding_size)
        self.user_mlp_embedding = nn.Embedding(self.num_users, self.mlp_embedding_size)
        self.item_mlp_embedding = nn.Embedding(self.num_services, self.mlp_embedding_size)
        self.mlp_layers = MLPLayers([2 * self.mlp_embedding_size] + self.mlp_hidden_size, self.dropout_prob)
        self.mlp_layers.logger = None  # remove logger to use torch.save()
        if self.gmf_train and self.mlp_train:
            self.predict_layer = nn.Linear(self.gmf_embedding_size + self.mlp_hidden_size[-1], 1)
        elif self.gmf_train:
            self.predict_layer = nn.Linear(self.gmf_embedding_size, 1)
        elif self.mlp_train:
            self.predict_layer = nn.Linear(self.mlp_hidden_size[-1], 1)

        # define loss
        self.loss = nn.L1Loss()

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            normal_(module.weight.data, mean=0.0, std=0.01)

    def forward(self, user, service):
        user_gmf_embedding = self.user_gmf_embedding(user)
        item_gmf_embedding = self.item_gmf_embedding(service)
        user_mlp_embedding = self.user_mlp_embedding(user)
        item_mlp_embedding = self.item_mlp_embedding(service)

        if self.gmf_train:
            gmf_output = torch.mul(user_gmf_embedding, item_gmf_embedding)
        if self.mlp_train:
            mlp_output = self.mlp_layers(torch.cat((user_mlp_embedding, item_mlp_embedding), -1))

        if self.gmf_train and self.mlp_train:
            output = self.predict_layer(torch.cat((gmf_output, mlp_output), -1))
        elif self.gmf_train:
            output = self.predict_layer(gmf_output)
        elif self.mlp_train:
            output = self.predict_layer(mlp_output)
        else:
            raise RuntimeError('mf_train and mlp_train can not be False at the same time')

        return output.squeeze(-1)

    def calculate_loss(self, user, service, rating):
        predict = self.forward(user, service)
        return self.loss(predict, rating)

    def predict(self, user, service):
        return self.forward(user, service)
