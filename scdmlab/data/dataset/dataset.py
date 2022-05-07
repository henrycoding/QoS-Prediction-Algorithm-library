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

    def _load_qos_matrix(self, dataset_type):
        """Load QoS matrix
        """
        matrix_path = os.path.join(self.dataset_path, f'{dataset_type}Matrix.txt')
        if not os.path.isfile(matrix_path):
            raise ValueError(f'File {matrix_path} not exist.')
        data = np.loadtxt(matrix_path)
        return data

    def _get_triad(self, matrix, nan_symbol=-1):
        """Converts matrix data to triad (uid,iid,rate)

        Args:
            nan_symbol (int, optional): The value used in the dataset to represent missing data. Defaults to -1.

        Returns:

        """
        triad_data = []
        matrix = deepcopy(matrix)
        matrix[matrix == nan_symbol] = 0
        non_zero_index_tuple = np.nonzero(matrix)
        for uid, sid in zip(non_zero_index_tuple[0], non_zero_index_tuple[1]):
            triad_data.append([uid, sid, matrix[uid, sid]])
        triad_data = np.array(triad_data)
        return triad_data

    def _split_train_test(self, data, density, shuffle=True):
        if shuffle:
            np.random.shuffle(data)

        train_num = int(len(data) * density)
        train_data, test_data = data[:train_num], data[train_num:]
        return train_data, test_data

    def build(self, *args):
        raise NotImplementedError('Method [next] should be implemented.')

    # TODO 无法保存
    def save(self):
        """Saving this `MatrixDataset` class
        """
        save_dir = self.config['checkpoint_dir']
        ensure_dir(save_dir)
        file = os.path.join(save_dir, f'{self.config["dataset"]}-dataset.pth')
        self.logger.info(set_color('Saving dataset into ', 'pink') + f'[{file}]')
        with open(file, 'wb') as f:
            pickle.dump(self, f)
