import os
from logging import getLogger

import numpy as np
import pandas as pd
from scdmlab.data.dataset import RT_MATRIX_DIR, TP_MATRIX_DIR, USER_DIR, SERVICE_DIR
from scdmlab.utils import set_color


# TODO 未重构
class DatasetBase(object):
    """
    指定要使用的数据集
    rt: rtMatrix
    tp: tpMatrix
    user: userlist
    service: wslist
    """

    def __init__(self, type_) -> None:
        super().__init__()

        self.type = type_
        assert self.type in ["rt", "tp", "user", "service"], f"类型不符，请在{['rt', 'tp', 'user', 'service']}中选择"

    def get_row_data(self):
        if self.type == "rt":
            data = np.loadtxt(RT_MATRIX_DIR)
        elif self.type == "tp":
            data = np.loadtxt(TP_MATRIX_DIR)
        elif self.type == "user":
            data = pd.read_csv(USER_DIR, sep="\t")
        elif self.type == "service":
            data = pd.read_csv(SERVICE_DIR, sep="\t")
        return data


class Dataset:
    def __int__(self, config):
        self.config = config
        self.dataset_name = config['dataset']
        self.dataset_path = self.config['data_path']
        # self.logger = getLogger()
        self._from_scratch()

    def _from_scratch(self):
        self.user_feat = pd.read_csv(os.path.join(self.dataset_path, 'userlist.txt'), sep="\t")
        self.service_feat = pd.read_csv(os.path.join(self.dataset_path, 'wslist.txt'), sep="\t")
        self.inter_rt = np.loadtxt(os.path.join(self.dataset_path, 'rtMatrix.txt'))
        self.inter_tp = np.loadtxt(os.path.join(self.dataset_path, 'tpMatrix.txt'))
