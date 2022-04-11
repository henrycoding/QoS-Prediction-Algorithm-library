import numpy as np
import torch
import torch.nn as nn

from scdmlab.utils import ModelType


class AbstractModel(nn.Module):
    def __init__(self):
        super(AbstractModel, self).__init__()

    def predict(self, interaction):
        raise NotImplementedError

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters' + f':{params}'


class GeneralModel(AbstractModel):
    type = ModelType.GENERAL

    def __init__(self, config, dataset):
        super(GeneralModel, self).__init__()

        self.device = config['device']
