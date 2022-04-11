import numpy as np
import pandas as pd
from scdmlab.data.dataset import RT_MATRIX_DIR, TP_MATRIX_DIR, USER_DIR, SERVICE_DIR


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
