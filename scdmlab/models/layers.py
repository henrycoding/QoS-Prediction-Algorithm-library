import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as fn
from torch.nn.init import normal_



def activation_layer(activation_name='relu'):
    if activation_name is None:
        activation = None
    elif isinstance(activation_name, str):
        if activation_name.lower() == 'sigmoid':
            activation = nn.Sigmoid()
        elif activation_name.lower() == 'tanh':
            activation = nn.Tanh()
        elif activation_name.lower() == 'relu':
            activation = nn.ReLU()
        elif activation_name.lower() == 'leakyrelu':
            activation = nn.LeakyReLU()
        elif activation_name.lower() == 'none':
            activation = None
    elif issubclass(activation_name, nn.Module):
        activation = activation_name()
    else:
        raise NotImplementedError(f"activation function {activation_name} is not implemented")

    return activation


class MLPLayers(nn.Module):
    def __init__(self, layers, drouput=0., activation='relu', bn=False, init_method=None):
        super(MLPLayers, self).__init__()
        self.layers = layers
        self.drouput = drouput
        self.activation = activation
        self.use_bn = bn
        self.init_method = init_method

        # mlp_modules = []
        mlp_modules = nn.ModuleList()
        for idx, (input_size, output_size) in enumerate(zip(self.layers[:-1], self.layers[1:])):
            mlp_modules.append(nn.Dropout(p=self.drouput))
            mlp_modules.append(nn.Linear(input_size, output_size))
            if self.use_bn:
                mlp_modules.append(nn.BatchNorm1d(num_features=output_size))
            activation_func = activation_layer(self.activation)
            if activation_func is not None:
                mlp_modules.append(activation_func)

        if self.init_method is not None:
            self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            if self.init_method == 'norm':
                normal_(module.weight.data, 0, 0.01)
            if module.bias is not None:
                module.bias.data.fill_(0.0)

    def forward(self, input_feature):
        return self.mlp_layers(input_feature)
