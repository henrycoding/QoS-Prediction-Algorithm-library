import os
import numpy as np
from time import time
from tqdm import tqdm
from logging import getLogger
import torch
import torch.optim as optim
from torch.nn.utils.clip_grad import clip_grad_norm_

from scdmlab.utils import get_local_time, set_color, get_gpu_usage, get_tensorboard
from scdmlab.evaluator.evaluation import mae, rmse


class AbstractTrainer(object):
    def __init__(self, config, model):
        self.config = config
        self.model = model

    def fit(self, train_data):
        raise NotImplementedError('Method [next] should be implemented.')

    def evaluate(self, eval_data):
        raise NotImplementedError('Method [next] should be implemented.')


class GeneralTrainer(AbstractTrainer):
    def __init__(self, config, model):
        super(GeneralTrainer, self).__init__(config, model)

        self.logger = getLogger()
        self.tensorboard = get_tensorboard(self.logger)
        self.gpu_available = torch.cuda.is_available() and config['use_gpu']
        self.device = config['device']
        self.checkpoint_dir = config['checkpoint_dir']
        saved_model_file = '{}-{}.pth'.format(self.config['model'], get_local_time())
        self.saved_model_file = os.path.join(self.checkpoint_dir, saved_model_file)
        self.train_loss_dict = dict()

        self.epochs = config['epochs']
        self.eval_step = min(config['eval_step'], self.epochs)
        self.stopping_step = config['stopping_step']
        self.clip_grad_norm = config['clip_grad_norm']

        # optimizer
        self.learner = config['learner']
        self.learning_rate = config['learning_rate']
        self.weight_decay = config['weight_decay']
        self.optimizer = self._build_optimizer()

        self.best_valid_result = np.inf

    def _build_optimizer(self, **kwargs):
        params = kwargs.get('params', self.model.parameters())
        learner = kwargs.get('learner', self.learner)
        learning_rate = kwargs.get('learning_rate', self.learning_rate)
        weight_decay = kwargs.get('weight_decay', self.weight_decay)

        if learner.lower() == 'adam':
            optimizer = optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
        else:
            self.logger.warning('Received unrecognized optimizer, set default Adam optimizer')
            optimizer = optim.Adam(params, lr=learning_rate)

        return optimizer

    def _train_epoch(self, train_data, epoch_idx, loss_func=None, show_progress=False):
        self.model.train()
        loss_func = loss_func or self.model.calculate_loss
        total_loss = None
        iter_data = (
            tqdm(
                train_data,
                total=len(train_data),
                ncols=100,
                desc=set_color(f"Train {epoch_idx:>5}", 'pink'),
            ) if show_progress else train_data
        )
        for batch_idx, batched_data in enumerate(iter_data):
            users, services, ratings = batched_data[0].to(self.device), \
                                       batched_data[1].to(self.device), \
                                       batched_data[2].to(self.device)
            self.optimizer.zero_grad()
            loss = loss_func(users, services, ratings)
            total_loss = loss.item() if total_loss is None else total_loss + loss.item()
            self._check_nan(loss)
            loss.backward()
            if self.clip_grad_norm:
                clip_grad_norm_(self.model.parameters(), **self.clip_grad_norm)
            self.optimizer.step()

            if self.gpu_available and show_progress:
                iter_data.set_postfix_str(set_color('GPU RAM: ' + get_gpu_usage(self.device), 'yellow'))

        return total_loss / len(train_data)

    @torch.no_grad()
    def _valid_epoch(self, epoch_idx, valid_data, show_progress=False):
        self.model.eval()
        loss_func = self.model.calculate_loss
        eval_total_loss = None
        iter_data = (
            tqdm(
                valid_data,
                total=len(valid_data),
                ncols=100,
                desc=set_color(f"Eval", 'pink'),
            ) if show_progress else valid_data
        )
        for batch_idx, batched_data in enumerate(iter_data):
            users, services, ratings = batched_data[0].to(self.device), \
                                       batched_data[1].to(self.device), \
                                       batched_data[2].to(self.device)
            loss = loss_func(users, services, ratings)
            eval_total_loss = loss.item() if eval_total_loss is None else eval_total_loss + loss.item()

        valid_result = eval_total_loss / len(valid_data)
        if valid_result < self.best_valid_result:
            self._save_checkpoint(epoch_idx)
            self.best_valid_result = valid_result

    def _save_checkpoint(self, epoch, verbose=True, **kwargs):
        """Store the model parameters information and training information.
        """
        saved_model_file = kwargs.pop('saved_model_file', self.saved_model_file)
        state = {
            'config': self.config,
            'epoch': epoch,
            # 'cur_step': self.cur_step,
            # 'best_valid_score': self.best_valid_score,
            'state_dict': self.model.state_dict(),
            # 'other_parameter': self.model.other_parameter(),
            'optimizer': self.optimizer.state_dict(),
        }
        torch.save(state, saved_model_file)
        if verbose:
            self.logger.info(set_color('Saving current', 'blue') + f': {saved_model_file}')

    def _check_nan(self, loss):
        if torch.isnan(loss):
            raise ValueError('Training loss is nan')

    def fit(self, train_data, valid_data=None, verbose=True, saved=True, show_progress=False):
        """Train the model based on the train data and the valid data.
        """
        for epoch_idx in range(self.epochs):
            train_loss = self._train_epoch(train_data, epoch_idx, show_progress=show_progress)
            self.train_loss_dict[epoch_idx] = train_loss
            self.logger.info(f"Training Epoch:[{epoch_idx + 1}/{self.epochs}] Loss:{train_loss:.4f}")

            if (epoch_idx + 1) % self.eval_step == 0:
                self._valid_epoch(epoch_idx, valid_data, show_progress)

    @torch.no_grad()
    def evaluate(self, eval_data, load_best_model=True, model_file=None, show_progress=False):
        if not eval_data:
            return

        if load_best_model:
            checkpoint_file = model_file or self.saved_model_file
            checkpoint = torch.load(checkpoint_file)
            self.model.load_state_dict(checkpoint['state_dict'])

        self.model.eval()
        iter_data = (
            tqdm(
                eval_data,
                total=len(eval_data),
                ncols=100,
                desc=set_color(f"Evaluate ", 'pink'),
            ) if show_progress else eval_data
        )
        predicts = []
        reals = []
        for batch_idx, batched_data in enumerate(iter_data):
            users, services, ratings = batched_data[0].to(self.device), \
                                       batched_data[1].to(self.device), \
                                       batched_data[2].to(self.device)
            y_pred = self.model.predict(users, services)
            predicts.extend(y_pred.tolist())
            reals.extend(ratings.tolist())

        result_mae = mae(reals, predicts)
        result_rmse = rmse(reals, predicts)
        self.logger.info(f"mae:{result_mae}, rmse:{result_rmse}")
