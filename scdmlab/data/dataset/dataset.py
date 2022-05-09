import os
import pickle
import numpy as np
import pandas as pd
from copy import deepcopy
from logging import getLogger

from scdmlab.utils import set_color, ensure_dir


class Dataset:
    def __init__(self, config):
        self.config = config
        self.dataset_name = config['dataset']
        self.dataset_path = config['data_path']
        self.logger = getLogger()

    def _get_triad(self, matrix, nan_symbol=-1):
        """Converts matrix data to triad (uid,iid,rate)

        Args:
            nan_symbol (int, optional): The value used in the dataset to represent missing data. Defaults to -1.
        """
        triad_data = []
        matrix = deepcopy(matrix)
        matrix[matrix == nan_symbol] = 0
        non_zero_index_tuple = np.nonzero(matrix)
        for uid, sid in zip(non_zero_index_tuple[0], non_zero_index_tuple[1]):
            triad_data.append([uid, sid, matrix[uid, sid]])
        triad_data = np.array(triad_data)
        return triad_data

    def _load_qos_matrix(self, dataset_type):
        """Load QoS matrix
        """
        matrix_path = os.path.join(self.dataset_path, f'{dataset_type}Matrix.txt')
        if not os.path.isfile(matrix_path):
            raise ValueError(f'File {matrix_path} not exist.')
        data = np.loadtxt(matrix_path)
        return data

    def _load_info_data(self, dataset_type):
        """Load Users/Services info dataset
        """
        if dataset_type == 'user':
            info_path = os.path.join(self.dataset_path, 'userlist.txt')
        elif dataset_type == 'service':
            info_path = os.path.join(self.dataset_path, 'wslist.txt')
        if not os.path.isfile(info_path):
            raise ValueError(f'File {info_path} not exist.')
        data = pd.read_csv(info_path, sep='\t')
        return data

    def _split_train_test(self, triad_data, density, shuffle=True):
        if shuffle:
            np.random.shuffle(triad_data)

        train_num = int(len(triad_data) * density)
        train_data, test_data = triad_data[:train_num], triad_data[train_num:]
        return train_data, test_data

    def build(self, *args):
        raise NotImplementedError('Method [build] should be implemented.')
