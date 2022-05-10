import os
import pickle
import numpy as np
import pandas as pd
from copy import deepcopy

from scdmlab.data import cache4method
from scdmlab.data.dataset import Dataset


class InfoDataset(Dataset):
    def __init__(self, config):
        super().__init__(config)
        self._from_scratch()

    def _fit(self, info_data, enabled_columns):
        feature2idx = {}
        feature2num = {}
        for column in enabled_columns:
            vc = info_data[column].value_counts(dropna=False)
            feature2idx[column] = {
                k: idx
                for idx, (k, v) in enumerate(vc.to_dict().items())
            }
            feature2num[column] = len(vc)
        return feature2idx, feature2num

    def _from_scratch(self):
        """Load dataset from scratch
        """
        self.rt_matrix = self._load_qos_matrix('rt')
        self.tp_matrix = self._load_qos_matrix('tp')
        self.NUM_USERS, self.NUM_SERVICES = self.rt_matrix.shape
        self.rt_traid_data = self._get_triad(self.rt_matrix)
        self.tp_traid_data = self._get_triad(self.tp_matrix)

        self.user_info_data = self._load_info_data('user')
        self.service_info_data = self._load_info_data('service')
        self.user_enable_columns = self.config['user_enable_columns']
        self.service_enable_columns = self.config['service_enable_columns']
        self.user_feature2idx, user_feature2num = self._fit(self.user_info_data, self.user_enable_columns)
        self.service_feature2idx, service_feature2num = self._fit(self.service_info_data, self.service_enable_columns)

    @cache4method
    def query_user(self, uid):
        row = self.user_info_data.iloc[uid, :]
        r = []
        for column in self.user_enable_columns:
            idx = self.user_feature2idx[column][row[column]]
            r.append(idx)
        return r

    @cache4method
    def query_service(self, sid):
        row = self.service_info_data.iloc[sid, :]
        r = []
        for column in self.service_enable_columns:
            idx = self.service_feature2idx[column][row[column]]
            r.append(idx)
        return r

    def _data_process(self, triad_data):
        r = []
        for row in triad_data:
            uid, sid, rating = int(row[0]), int(row[1]), float(row[2])
            user_info = self.query_user(uid)
            service_info = self.query_service(sid)
            r.append([user_info, service_info, rating])
        return r

    def build(self, density, dataset_type):
        if dataset_type == 'rt':
            info_triad_data = self._data_process(self.rt_traid_data)
            train_data, test_data = self._split_train_test(info_triad_data, density)
        elif dataset_type == 'tp':
            info_triad_data = self._data_process(self.tp_traid_data)
            train_data, test_data = self._split_train_test(info_triad_data, density)
        else:
            raise ValueError(f'{dataset_type} not support')

        return train_data, test_data
