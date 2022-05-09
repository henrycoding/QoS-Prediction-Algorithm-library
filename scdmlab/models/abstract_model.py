import numpy as np
from logging import getLogger
import torch
import torch.nn as nn

from scdmlab.utils import ModelType


class AbstractModel(nn.Module):
    """Base class for all models
    """

    def __init__(self):
        self.logger = getLogger()
        super(AbstractModel, self).__init__()

    def calculate_loss(self, **kwargs):
        raise NotImplementedError

    def predict(self, **kwargs):
        raise NotImplementedError

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters' + f':{params}'


class GeneralModel(AbstractModel):
    model_type = ModelType.GENERAL

    def __init__(self, config, dataset):
        super(GeneralModel, self).__init__()
        self.num_users = dataset.NUM_USERS
        self.num_services = dataset.NUM_SERVICES
        self.device = config['device']
