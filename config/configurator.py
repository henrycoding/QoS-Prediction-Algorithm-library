import re
import os
import sys
import yaml
import torch

from utils import get_model, general_arguments, training_arguments, evaluation_arguments, dataset_arguments


class Config(object):
    def __init__(self, model=None, dataset=None):
        self._init_parameters_category()

        self.model, self.model_class, self.dataset = self._get_model_and_dataset(model, dataset)

    def _init_parameters_category(self):
        self.parameters = dict()
        self.parameters['General'] = general_arguments
        self.parameters['Training'] = training_arguments
        self.parameters['Evaluation'] = evaluation_arguments
        self.parameters['Dataset'] = dataset_arguments

    def _get_model_and_dataset(self, model, dataset):
        if not isinstance(model, str):
            final_model_class = model
            final_model = model.__name__
        else:
            final_model = model
            final_model_class = get_model(final_model)

        return final_model, final_model_class, dataset
