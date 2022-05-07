import os
import pickle
import numpy as np
import pandas as pd
from copy import deepcopy
from scdmlab.data.dataset import Dataset
from scdmlab.utils import set_color, ensure_dir


class MatrixDataset(Dataset):
    def __init__(self, config):
        super().__init__(config)
        self._from_scratch()

    def _from_scratch(self):
        """Load dataset from scratch.
        Initialize attributes firstly, then load data from atomic files, pre-process the dataset lastly.
        """
        self.logger.debug(set_color(f'Loading {self.__class__} from scratch.', 'green'))
        self.rt_matrix = self._load_qos_matrix('rt')
        self.tp_Matrix = self._load_qos_matrix('tp')
        self.row_num, self.col_num = self.rt_matrix.shape

    def build(self, density, dataset_type):
        if dataset_type == 'rt':
            data = self._get_triad(self.rt_matrix)
        elif dataset_type == 'tp':
            data = self._get_triad(self.tp_Matrix)
        else:
            raise ValueError(f'{dataset_type} not support')

        train_data, test_data = self._split_train_test(data, density)
        return train_data, test_data
