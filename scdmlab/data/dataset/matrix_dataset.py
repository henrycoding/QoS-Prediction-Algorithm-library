import numpy as np
import pandas as pd
from copy import deepcopy
from scdmlab.data.dataset import DatasetBase


# TODO 未重构
class MatrixDataset(DatasetBase):
    def __init__(self, type_) -> None:
        super().__init__(type_)
        assert type_ in ["rt", "tp"], f"类型不符，请在{['rt', 'tp']}中选择"
        self.matrix = self._get_row_data()
        self.scaler = None

    # def get_similarity_matrix(self, method="cos"):
    #     assert len(self.matrix) != 0, "matrix should not be empty"
    #     similarity_matrix = None
    #     if method == "cos":
    #         similarity_matrix = cosine_similarity(self.matrix)
    #     elif method == "pcc":
    #         ...
    #     return similarity_matrix

    def _get_row_data(self):
        data = super().get_row_data()
        if isinstance(data, pd.DataFrame):
            data = data.to_numpy()
        self.row_n, self.col_n = data.shape
        return data

    def get_triad(self, nan_symbol=-1):
        """生成三元组(uid,iid,rate)

        Args:
            nan_symbol (int, optional): 数据集中用于表示数据缺失的值. Defaults to -1.

        Returns:
            list[list]: (uid,iid,rate)
        """
        triad_data = []
        row_data = deepcopy(self.matrix)

        row_data[row_data == nan_symbol] = 0
        non_zero_index_tuple = np.nonzero(row_data)
        for uid, iid in zip(non_zero_index_tuple[0], non_zero_index_tuple[1]):
            triad_data.append([uid, iid, row_data[uid, iid]])
        triad_data = np.array(triad_data)
        print("triad_data size:", triad_data.shape)
        return triad_data

    def split_train_test(self,
                         density,
                         nan_symbol=-1,
                         shuffle=True,
                         normalize_type=None):
        triad_data = self.get_triad(nan_symbol)

        if shuffle:
            np.random.shuffle(triad_data)

        train_n = int(self.row_n * self.col_n * density)  # 训练集数量
        train_data, test_data = triad_data[:train_n], triad_data[train_n:]
        # if normalize_type is not None:
        #     self.__norm_train_test_data(train_data, test_data, normalize_type)

        return train_data, test_data

    # def __norm_train_test_data(self,
    #                            train_data,
    #                            test_data,
    #                            scaler_type="z_score"):
    #     if scaler_type == "z_score":
    #         f = z_score
    #     elif scaler_type == "l2_norm":
    #         f = l2_norm
    #     elif scaler_type == "min_max":
    #         f = min_max_scaler
    #     else:
    #         raise NotImplementedError
    #     x_train, scaler = f(train_data)
    #     x_test, scaler = f(test_data, scaler)
    #     self.scaler = scaler
    #     train_data[:, 2] = x_train[:, 2]
    #     test_data[:, 2] = x_test[:, 2]
    #
    # def get_mini_triad(self, nan_symbol=-1, sample_nums=200):
    #     total_triad_data = self.get_triad(nan_symbol)
    #     return random.sample(total_triad_data, sample_nums)
    #
    # def mini_split_train_test(self, density, nan_symbol=-1, shuffle=True):
    #
    #     triad_data = self.get_mini_triad(nan_symbol)
    #
    #     if shuffle:
    #         np.random.shuffle(triad_data)
    #
    #     train_n = int(self.row_n * self.col_n * density)  # 训练集数量
    #     train_data, test_data = triad_data[:train_n, :], triad_data[
    #                                                      train_n:, :]
    #
    #     return train_data, test_data
