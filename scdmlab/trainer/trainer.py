import os
import numpy as np
from time import time
from tqdm import tqdm
from logging import getLogger
import torch
import torch.optim as optim
from torch.nn.utils.clip_grad import clip_grad_norm_

from scdmlab.utils import get_local_time, set_color, get_gpu_usage


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

        self.epochs = config['epochs']
        self.eval_step = min(config['eval_step'], self.epochs)
        self.clip_grad_norm = config['clip_grad_norm']

        # optimizer
        self.learner = config['learner']
        self.learning_rate = config['learning_rate']
        self.weight_decay = config['weight_decay']
        self.optimizer = self._build_optimizer()

        self.gpu_available = torch.cuda.is_available() and config['use_gpu']
        self.device = config['device']
        self.checkpoint_dir = config['checkpoint_dir']

        saved_model_file = '{}-{}.pth'.format(self.config['model'], get_local_time())
        self.saved_model_file = os.path.join(self.checkpoint_dir, saved_model_file)
        self.train_loss_dict = dict()

    def _build_optimizer(self, **kwargs):
        params = kwargs.pop('params', self.model.parameters())
        learner = kwargs.pop('learner', self.learner)
        learning_rate = kwargs.pop('learning_rate', self.learning_rate)
        weight_decay = kwargs.pop('weight_decay', self.weight_decay)

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
            users, services, ratings = batched_data[0].to(self.device), batched_data[1].to(self.device), batched_data[
                2].to(self.device)
            self.optimizer.zero_grad()
            losses = loss_func(users, services, ratings)
            if isinstance(losses, tuple):
                # TODO
                pass
            else:
                loss = losses
                total_loss = losses.item() if total_loss is None else total_loss + losses.item()
            self._check_nan(loss)
            loss.backward()
            # TODO 梯度裁剪
            if self.clip_grad_norm:
                clip_grad_norm_(self.model.parameters(), **self.clip_grad_norm)
            self.optimizer.step()
            if self.gpu_available and show_progress:
                iter_data.set_postfix_str(set_color('GPU RAM: ' + get_gpu_usage(self.device), 'yellow'))
        return total_loss

    def _check_nan(self, loss):
        if torch.isnan(loss):
            raise ValueError('Training loss is nan')

    def fit(self, train_data, valid_data=None, verbose=True, saved=True, show_progress=False):
        """Train the model based on the train data and the valid data.

        Args:
            train_data (DataLoader): the train data
            valid_data (DataLoader, optional): the valid data, default: None.
                                               If it's None, the early_stopping is invalid.
            verbose (bool, optional): whether to write training and evaluation information to logger, default: True
            saved (bool, optional): whether to save the model parameters, default: True
            show_progress (bool): Show the progress of training epoch and evaluate epoch. Defaults to ``False``.
                                    Includes (epoch_idx, valid_score) input arguments.

        Returns:
             (float, dict): best valid score and best valid result. If valid_data is None, it returns (-1, None)
        """

        for epoch_idx in range(self.epochs):
            train_loss = self._train_epoch(train_data, epoch_idx, show_progress=show_progress)
            self.train_loss_dict[epoch_idx] = sum(train_loss) if isinstance(train_loss, tuple) else train_loss

    @torch.no_grad()
    def evaluate(self, eval_data, load_best_model=True, model_file=None, show_progress=False):
        if not eval_data:
            return

        if load_best_model:
            checkpoint_file = model_file or self.saved_model_file
            checkpoint = torch.load(checkpoint_file)
            self.model.load_state_dict(checkpoint['state_dict'])

        self.model.eval()
