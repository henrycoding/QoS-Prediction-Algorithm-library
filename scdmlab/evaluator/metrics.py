import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

from scdmlab.evaluator import LossMetric


class MAE(LossMetric):
    def __init__(self, config):
        super().__init__(config)

    def calculate_metric(self, dataobject):
        return self.output_metric('mae', dataobject)

    def metric_info(self, preds, trues):
        return mean_absolute_error(trues, preds)


class RMSE(LossMetric):
    def __init__(self, config):
        super().__init__(config)

    def calculate_metric(self, dataobject):
        return self.output_metric('rmse', dataobject)

    def metric_info(self, preds, trues):
        return np.sqrt(mean_squared_error(trues, preds))
