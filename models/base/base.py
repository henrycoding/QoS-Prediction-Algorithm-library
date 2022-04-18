import os
import sys
import time
import torch
from tensorboard import program
from tqdm import tqdm
from yacs.config import CfgNode

from utils.LoadModelData import send_train_progress
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


class ModelBase(object):
    def __init__(self, model, config: CfgNode, writer: SummaryWriter):
        self.model = model
        self.config = config

        # model name
        try:
            self.model_name = self.config.MODEL.NAME
        except Exception:
            raise Exception("The model name is not provided in the configuration file!")

        # device
        self.device = get_device(self.config)
        # model save name
        try:
            self.model_save_name = self.config.MODEL.SAVE_NAME
        except:
            self.model_save_name = ''

        # density
        try:
            self.density = self.config.TRAIN.DENSITY
        except Exception:
            raise Exception("The parameter 'TRAIN.DENSITY' not found!")

        # loss function
        self.loss_fn = use_loss_fn(self.config)

        # optimizer
        self.opt = use_optimizer(self.model, self.config)

        # log
        self.logger = TNLog(self.model_name)
        self.logger.initial_logger()

        # running time
        self.date = self.config.MODEL.DATE

        # TensorBoard
        if writer is None:
            self._writer = SummaryWriter()
        else:
            self._writer = writer
            self._tensorboard_title = f"Density_{self.config.TRAIN.DENSITY}"

        # save model parameters
        self.saved_model_ckpt = []

        # best loss
        self.best_loss = None

    def _evaluate(self, eval_loader, epoch, eval_loss_list):
        assert eval_loader is not None, "Please offer eval dataloader"

        self.model.eval()
        with torch.no_grad():
            eval_total_loss = 0
            for batch in tqdm(eval_loader, desc='Evaluating', position=0):
                user, item, rating = batch[0].to(self.device), \
                                     batch[1].to(self.device), \
                                     batch[2].to(self.device)
                y_pred = self.model(user, item)
                y_real = rating.reshape(-1, 1)
                loss = self.loss_fn(y_pred, y_real)
                eval_total_loss += loss.item()

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

            self.logger.info(f"Test loss: {loss_per_epoch:.4f}")
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
            # save_filename = f"density_{self.density}_loss_{self.best_loss:.4f}{_{self.model_save_name}" if len(self.model_save_name) else ''}.ckpt"
            save_filename = str(f"density_{self.density}_loss_{self.best_loss:.4f}") + str(
                f"_{self.model_save_name}" if len(self.model_save_name) else '') + '.ckpt'
            save_checkpoint(ckpt, is_best, save_dirname, save_filename)

    def fit(self, train_loader, eval_loader):
        self.model.train()
        self.model.to(self.device)

        train_loss_list = []
        eval_loss_list = []

        # training
        try:
            num_epochs = self.config.TRAIN.NUM_EPOCHS
        except:
            num_epochs = 200

        # tqdm
        for epoch in tqdm(range(num_epochs), desc=f'Training Density={self.density}'):
            progress = int(epoch / num_epochs * 100)
            send_train_progress(progress)
            train_batch_loss = 0
            for batch in train_loader:
                users, items, ratings = batch[0].to(self.device), \
                                        batch[1].to(self.device), \
                                        batch[2].to(self.device)
                self.opt.zero_grad()
                y_real = ratings.reshape(-1, 1)
                y_pred = self.model(users, items)
                loss = self.loss_fn(y_pred, y_real)
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
        y_pred_list = []
        y_list = []

        # try:
        #     resume = self.config.TRAIN.PRETRAIN
        # except:
        #     resume = False
        # else:
        #     try:
        #         path = self.config.TRAIN.PRETRAIN_DIR
        #     except Exception:
        #         raise Exception("The 'TRAIN.PRETRAIN_DIR' is not provided in the configuration file!")

        # load the pre-training model
        # if resume:
        #     ckpt = load_checkpoint(path)
        # else:
        # select the model with the least loss
        models = sorted(self.saved_model_ckpt, key=lambda x: x['best_loss'])
        ckpt = models[0]

        self.model.load_state_dict(ckpt['model'])
        self.logger.info(f"last checkpoint restored! ckpt: loss {ckpt['best_loss']:.4f} Epoch {ckpt['epoch']}")

        self.model.to(self.device)
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(test_loader, position=0):
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

        return torch.cat(y_list).cpu().numpy(), torch.cat(y_pred_list).cpu().numpy()


class MemoryBase(object):
    def __init__(self) -> None:
        super().__init__()

    def fit(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError
