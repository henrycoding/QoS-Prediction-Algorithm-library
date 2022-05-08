class AbstractMetric(object):
    def __init__(self, config):
        self.decimal_place = config['metric_decimal_place']

    def calculate_metric(self, dataobject):
        raise NotImplementedError('Method [calculate_metric] should be implemented.')


class LossMetric(AbstractMetric):
    def __init__(self, config):
        super().__init__(config)

    def used_info(self, dataobject):
        preds = dataobject.get('model.score')
        trues = dataobject.get('data.label')

        return preds.squeeze(-1).numpy(), trues.squeeze(-1).numpy()

    def output_metric(self, metric, dataobject):
        preds, trues = self.used_info(dataobject)
        result = self.metric_info(preds, trues)
        return {metric: round(result, self.decimal_place)}

    def metric_info(self, preds, trues):
        raise NotImplementedError('Method [metric_info] of loss-based metric should be implemented.')
