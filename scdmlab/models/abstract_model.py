import numpy as np
from logging import getLogger
import torch
import torch.nn as nn

from scdmlab.utils import ModelType


class AbstractModel(nn.Module):
    def __init__(self):
        self.logger = getLogger()
        super(AbstractModel, self).__init__()

    def calculate_loss(self, *args):
        raise NotImplementedError

    def predict(self, *args):
        raise NotImplementedError

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters' + f':{params}'


class GeneralModel(AbstractModel):
    type = ModelType.GENERAL

    def __init__(self, config, dataset):
        super(GeneralModel, self).__init__()
        self.num_users = dataset.row_num
        self.num_services = dataset.col_num
        self.device = config['device']
