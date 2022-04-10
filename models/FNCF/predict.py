import numpy as np
from collections import defaultdict

import pandas as pd
import torch
from tqdm import tqdm
from yacs.config import CfgNode

from models.FNCF.model import NeuMF
from models.FNCF.config import get_cfg_defaults
from utils.LoadModelData import set_model_result
from utils.model_util import data_loading
import os
# evaluation indicator
from utils.evaluation import mae, mse, rmse


class Predict:
    def __init__(self, config: CfgNode, parameters) -> None:
        self.res = {}
        self.result_show = None
        self.result = defaultdict(list)
        self.model = None
        self.device = torch.device("cuda")

        # config
        self.config = config
        # self.data_type = config.TRAIN.DATA_TYPE
        # self.density_list = config.TRAIN.DENSITY_LIST
        # self.batch_size = config.TRAIN.BATCH_SIZE
        # self.model_load = absolute(config.MODEL.LOAD_PATH)

        # parameters
        self.density = parameters['density']
        self.load_path = parameters['model']['savePath']
        self.data_type = parameters['dataset']
        self.batch_size = parameters['batchSize']

    def predict(self):
        for density in self.density_list:
            rt_data, train_dataloader, test_dataloader = data_loading(self.data_type, density, self.batch_size)
            num_users = rt_data.row_n
            num_items = rt_data.col_n
            self.config.defrost()
            self.config.TRAIN.NUM_USERS = num_users
            self.config.TRAIN.NUM_ITEMS = num_items
            self.config.TRAIN.DENSITY = density
            self.config.freeze()
            path = os.path.join(self.model_load, "Density_{}".format(density))
            dirs = os.listdir(path)
            model = NeuMF(self.config)
            self.model = model
            best_model_path = os.path.join(path, dirs[0])
            self.model.load_state_dict(torch.load(best_model_path)['model'])
            self.run(test_dataloader, density)
        self.result_show = pd.DataFrame(self.result, index=['MAE', 'MSE', 'RMSE'])
        print(self.result_show)

    def predict_one(self):
        # load_path = "C:/Users/Administrator/Desktop/test/QoS-Predcition-Algorithm-library/output/NeuMF/2022-04-07_13-52-20/saved_model/Density_0.2/density_0.2_loss_0.5712.ckpt"
        rt_data, train_dataloader, test_dataloader = data_loading(self.data_type, self.density, self.batch_size)
        num_users = rt_data.row_n
        num_items = rt_data.col_n
        self.config.defrost()
        self.config.TRAIN.NUM_USERS = num_users
        self.config.TRAIN.NUM_ITEMS = num_items
        self.config.freeze()
        model = NeuMF(self.config)
        model.load_state_dict(torch.load(self.load_path)['model'])
        self.model = model
        self.run(test_dataloader, self.density)
        self.result_show = pd.DataFrame(self.result, index=['MAE', 'MSE', 'RMSE'])
        print(self.result_show)

    def run(self, test_loader, density):
        y_pred_list = []
        y_list = []
        self.model.to(self.device)
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f'Predicting Density={density:.2f}', position=0):
                user, item, rating = batch[0].to(self.device), \
                                     batch[1].to(self.device), \
                                     batch[2].to(self.device)
                y_pred = self.model(user, item).squeeze()
                y_real = rating.reshape(-1, 1)
                if len(y_pred.shape) == 0:  # 防止因batch大小而变成标量,故增加一个维度
                    y_pred = y_pred.unsqueeze(dim=0)
                if len(y_real.shape) == 0:  # 防止因batch大小而变成标量,故增加一个维度
                    y_real = y_real.unsqueeze(dim=0)
                y_pred_list.append(y_pred)
                y_list.append(y_real)
        y, y_pred = torch.cat(y_list).cpu().numpy(), torch.cat(y_pred_list).cpu().numpy()
        mae_ = mae(y, y_pred)
        mse_ = mse(y, y_pred)
        rmse_ = rmse(y, y_pred)
        self.res = {
            'mae': np.float(mae_),
            'mse': np.float(mse_),
            'rmse': np.float(rmse_),
        }
        self.result[density].extend([mae_, mse_, rmse_])


def start_predict(parameters):
    print("start_predict")
    cfg = get_cfg_defaults()
    predict = Predict(cfg, parameters)
    predict.predict_one()
    set_model_result(parameters['id'], predict.res)


if __name__ == '__main__':
    cfg = get_cfg_defaults()
    predict = Predict(cfg)
    predict.predict_one()

"""
          0.05      0.10      0.15      0.20
MAE   0.582943  0.595642  0.580388  0.571211
MSE   3.598602  3.594801  3.588758  3.601026
RMSE  1.896998  1.895996  1.894402  1.897637

          0.05      0.10      0.15      0.20
MAE   0.581500  0.594697  0.579646  0.569527
MSE   3.599760  3.594886  3.586206  3.599837
RMSE  1.897303  1.896018  1.893728  1.897324
"""
