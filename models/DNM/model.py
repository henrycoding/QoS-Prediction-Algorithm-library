import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from models.DNM.config import get_cfg_defaults
from models.base import ModelBase
from utils.model_util import ModelTest
from copy import deepcopy


class DNM(nn.Module):
    def __init__(self, config):
        super(DNM, self).__init__()
        self.config = config

        self.embedding_dim = config.embedding_dim

        self.UID_embedding = nn.Embedding(339, self.embedding_dim)
        self.UR_embedding = nn.Embedding(31, self.embedding_dim)
        self.UAS_embedding = nn.Embedding(137, self.embedding_dim)
        self.SID_embedding = nn.Embedding(5825, self.embedding_dim)
        self.SR_embedding = nn.Embedding(74, self.embedding_dim)
        self.SAS_embedding = nn.Embedding(993, self.embedding_dim)

        self.dropout = nn.Dropout(p=0.2)

        self.perception_layers = nn.ModuleList()
        layers = self.config.perception_layers
        for in_size, out_size in zip(layers[:-1], layers[1:]):
            self.perception_layers.append(nn.Linear(in_size, out_size))
            self.perception_layers.append(nn.BatchNorm1d(out_size))

        self.task_specific_layers_rt = nn.ModuleList()
        layers = self.config.task_specific_layers_rt
        for in_size, out_size in zip(layers[:-1], layers[1:]):
            self.task_specific_layers_rt.append(nn.Linear(in_size, out_size))
            self.task_specific_layers_rt.append(nn.BatchNorm1d(out_size))
        self.affine_output_rt = nn.Linear(layers[-1], 1)
        self.logistic_rt = nn.Sigmoid()

        self.task_specific_layers_tp = nn.ModuleList()
        layers = self.config.task_specific_layers_tp
        for in_size, out_size in zip(layers[:-1], layers[1:]):
            self.task_specific_layers_tp.append(nn.Linear(in_size, out_size))
            self.task_specific_layers_tp.append(nn.BatchNorm1d(out_size))
        self.affine_output_tp = nn.Linear(layers[-1], 1)
        self.logistic_tp = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight)

    def forward(self, user_indices, service_indices):
        uid_embedding = self.UID_embedding(user_indices[:, 0])
        ur_embedding = self.UR_embedding(user_indices[:, 1])
        uas_embedding = self.UAS_embedding(user_indices[:, 2])
        sid_embedding = self.SID_embedding(service_indices[:, 0])
        sr_embedding = self.SR_embedding(service_indices[:, 1])
        sas_embedding = self.SAS_embedding(service_indices[:, 2])

        x_embed = [uid_embedding, ur_embedding, uas_embedding, sid_embedding, sr_embedding, sas_embedding]

        product_vec = None
        for i in range(0, 3):
            for j in range(3, 6):
                vec = torch.mul(x_embed[i], x_embed[j])
                if product_vec is None:
                    product_vec = vec
                else:
                    product_vec = product_vec + vec

        add_vec = None
        for i in range(0, 6):
            if add_vec is None:
                add_vec = x_embed[i]
            else:
                add_vec = add_vec + x_embed[i]

        vector = torch.cat([product_vec, add_vec], dim=-1)
        vector = self.dropout(vector)

        for layer in self.perception_layers:
            vector = layer(vector)
            vector = F.relu(vector)

        vector_rt = vector
        for layer in self.task_specific_layers_rt:
            vector_rt = layer(vector_rt)
            vector_rt = F.relu(vector_rt)
        logits_rt = self.affine_output_rt(vector_rt)
        rating_rt = self.logistic_rt(logits_rt)

        vector_tp = vector
        for layer in self.task_specific_layers_tp:
            vector_tp = layer(vector_tp)
            vector_tp = F.relu(vector_tp)
        logits_tp = self.affine_output_tp(vector_tp)
        rating_tp = self.logistic_tp(logits_tp)

        results = torch.cat([rating_rt, rating_tp], dim=-1)
        return results


class DNMModel(ModelBase):
    def __init__(self, config, writer=None):
        model = DNM(config)
        super(DNMModel, self).__init__(model, config, writer)

    def parameters(self):
        return self.model.parameters()

    def __str__(self) -> str:
        return str(self.model)

    def __repr__(self) -> str:
        return repr(self.model)


if __name__ == "__main__":
    cfg = get_cfg_defaults()
    test = ModelTest(DNMModel, cfg)
    test.run()
