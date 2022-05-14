import os
import pickle
import numpy as np
import pandas as pd
from copy import deepcopy
from scdmlab.data.dataset import AbstractDataset


class MatrixDataset(AbstractDataset):
    def __init__(self, config):
        super().__init__(config)
        self._from_scratch()

    def _from_scratch(self):
        """Load dataset from scratch
        """
        self.rt_matrix = self._load_qos_matrix('rt')
        self.tp_matrix = self._load_qos_matrix('tp')
        self.NUM_USERS, self.NUM_SERVICES = self.rt_matrix.shape

        self.rt_traid_data = self._get_triad(self.rt_matrix)
        self.tp_traid_data = self._get_triad(self.tp_matrix)

    def build(self, density, dataset_type):
        if dataset_type == 'rt':
            train_data, test_data = self._split_train_test(self.rt_traid_data, density)
        elif dataset_type == 'tp':
            train_data, test_data = self._split_train_test(self.tp_traid_data, density)
        else:
            raise ValueError(f'{dataset_type} not support')

        return train_data, test_data
