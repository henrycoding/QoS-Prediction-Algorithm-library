import os
import sys
import time
import torch
from tensorboard import program
from tqdm import tqdm
from yacs.config import CfgNode
from utils.model_util import get_device

from root import absolute
from utils.evaluation import mae, mse, rmse

from .utils import train_single_epoch_with_dataloader, train_mult_epochs_with_dataloader
# 日志
from utils.mylogger import TNLog

# 模型保存
from utils.model_util import save_checkpoint, load_checkpoint, use_optimizer, use_loss_fn

# TensorBoard
from torch.utils.tensorboard import SummaryWriter

from utils.request import send_persentage


class ModelBase(object):
    def __init__(self, model, config, writer):
        self.model = model
        self.config = config

        # model name
        self.model_name = self.config.model_name

        # device
        self.device = get_device(self.config)

        # density
        self.density = self.config.density

        # loss function
        self.loss_fn = use_loss_fn(self.config.loss_fn)

        # optimizer
        self.opt = use_optimizer(self.model, self.config)

        # log
        self.logger = TNLog(self.model_name)
        self.logger.initial_logger()

        # running time
        self.date = self.config.date

        # TensorBoard
        if writer is None:
            self._writer = SummaryWriter()
        else:
            self._writer = writer
            self._tensorboard_title = f"Density_{self.config.density}"

        # save model parameters
        self.saved_model_ckpt = []

        # best loss
        self.best_loss = None

    def _evaluate(self, eval_loader, epoch, eval_loss_list):
        assert eval_loader is not None, "Please offer eval dataloader"

        self.model.eval()
        with torch.no_grad():
            eval_total_loss_rt = 0
            eval_total_loss_tp = 0
            eval_total_loss = 0
            for batch in tqdm(eval_loader, desc='Evaluating', position=0):
                user, item, rating = batch[0].to(self.device), \
                                     batch[1].to(self.device), \
                                     batch[2].to(self.device)
                y_real = rating.reshape(-1, 2)
                y_pred = self.model(user, item)
                loss_rt = self.loss_fn(y_pred[:, 0], y_real[:, 0])
                loss_tp = self.loss_fn(y_pred[:, 1], y_real[:, 1])
                loss = 0.95 * loss_rt + 0.05 * loss_tp
                eval_total_loss_rt += loss_rt.item()
                eval_total_loss_tp += loss_tp.item()
                eval_total_loss += loss.item()

            loss_per_epoch_rt = eval_total_loss_rt / len(eval_loader)
            loss_per_epoch_tp = eval_total_loss_tp / len(eval_loader)
            loss_per_epoch = eval_total_loss / len(eval_loader)

            if self.best_loss is None:
                self.best_loss = loss_per_epoch
                is_best = True
            elif loss_per_epoch < self.best_loss:
                self.best_loss = loss_per_epoch
                is_best = True
            else:
                is_best = False

            eval_loss_list.append(loss_per_epoch)

            self.logger.info(f"Test loss (rt): {loss_per_epoch_rt:.4f}")
            self.logger.info(f"Test loss (tp): {loss_per_epoch_tp:.4f}")
            self.logger.info(f"Test loss: {loss_per_epoch:.4f}")
            self._writer.add_scalar(f"{self._tensorboard_title}/Eval loss (rt)", loss_per_epoch_rt, epoch + 1)
            self._writer.add_scalar(f"{self._tensorboard_title}/Eval loss (tp)", loss_per_epoch_tp, epoch + 1)
            self._writer.add_scalar(f"{self._tensorboard_title}/Eval loss", loss_per_epoch, epoch + 1)

            # 保存最优的loss
            ckpt = {}
            if is_best:
                ckpt = {
                    "model": self.model.state_dict(),
                    "epoch": epoch + 1,
                    "optim": self.opt.state_dict(),
                    "best_loss": self.best_loss
                }
                self.saved_model_ckpt.append(ckpt)
            save_dirname = f"output/{self.model_name}/{self.date}/saved_model/Density_{self.density}"
            save_filename = f"density_{self.density}_loss_{self.best_loss:.4f}.ckpt"
            save_checkpoint(ckpt, is_best, save_dirname, save_filename)

    def fit(self, train_loader, eval_loader):
        self.model.train()
        self.model.to(self.device)

        train_loss_list = []
        eval_loss_list = []

        # training
        num_epochs = self.config.num_epochs

        for epoch in tqdm(range(num_epochs), desc=f'Training Density={self.density}'):
            train_batch_loss = 0

            for batch in train_loader:
                users, items, ratings = batch[0].to(self.device), \
                                        batch[1].to(self.device), \
                                        batch[2].to(self.device)
                self.opt.zero_grad()
                y_real = ratings.reshape(-1, 2)
                y_pred = self.model(users, items)
                loss_rt = self.loss_fn(y_pred[:, 0], y_real[:, 0])
                loss_tp = self.loss_fn(y_pred[:, 1], y_real[:, 1])
                loss = 0.95 * loss_rt + 0.05 * loss_tp
                loss.backward()
                self.opt.step()

                train_batch_loss += loss.item()

            loss_per_epoch = train_batch_loss / len(train_loader)
            train_loss_list.append(loss_per_epoch)

            self.logger.info(f"Training Epoch:[{epoch + 1}/{num_epochs}] Loss:{loss_per_epoch:.4f}")
            self._writer.add_scalar(f"{self._tensorboard_title}/Train loss", loss_per_epoch, epoch + 1)

            # 验证
            if (epoch + 1) % 10 == 0:
                self._evaluate(eval_loader, epoch, eval_loss_list)

    # 预测
    def predict(self, test_loader):
        # select the model with the least loss
        ckpt = self.saved_model_ckpt[-1]

        self.model.load_state_dict(ckpt['model'])
        self.logger.info(f"last checkpoint restored! ckpt: loss {ckpt['best_loss']:.4f} Epoch {ckpt['epoch']}")

        self.model.to(self.device)
        self.model.eval()

        real, pred = {}, {}
        for i in range(2):
            real[i], pred[i] = list(), list()
        with torch.no_grad():
            for batch in tqdm(test_loader, position=0):
                user, item, rating = batch[0].to(self.device), \
                                     batch[1].to(self.device), \
                                     batch[2].to(self.device)
                y_real = rating.reshape(-1, 2)
                y_pred = self.model(user, item)
                for i in range(2):
                    real[i].extend(y_real[:, i].tolist())
                    pred[i].extend(y_pred[:, i].tolist())

        return real, pred


class MemoryBase(object):
    def __init__(self) -> None:
        super().__init__()

    def fit(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError
