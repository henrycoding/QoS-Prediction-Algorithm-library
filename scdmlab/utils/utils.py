import os
import random
import datetime
import importlib
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from scdmlab.utils.enum_type import ModelType


def get_local_time():
    r"""Get current time

    Returns:
        str: current time
    """
    cur = datetime.datetime.now()
    cur = cur.strftime('%b-%d-%Y_%H-%M-%S')

    return cur


def ensure_dir(dir_path):
    """Make sure the directory exists, if it does not exist, create it

    Args:
        dir_path (str): directory path

    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def get_model(model_name: str):
    """Automatically select model class based on model name

    Args:
        model_name (str): model name
    """
    model_submodule = [
        'general_model'
    ]

    model_module = None
    for submodule in model_submodule:
        module_path = '.'.join(['scdmlab.models', submodule, model_name])
        if importlib.util.find_spec(module_path, __name__):
            model_module = importlib.import_module(module_path, __name__)
            break

    if model_module is None:
        raise ValueError(f'`model_name` [{model_name}] is not the name of an existing model.')

    model_class = getattr(model_module, model_name)
    return model_class


def get_trainer(model_type, model_name):
    try:
        return getattr(importlib.import_module('scdmlab.trainer'), model_name + 'Trainer')
    except AttributeError:
        if model_type == ModelType.GENERAL:
            return getattr(importlib.import_module('scdmlab.trainer'), 'GeneralTrainer')


def init_seed(seed: int, reproducibility: bool) -> None:
    """init random seed for random functions in numpy, torch, cuda and cudnn

    Args:
        seed (int): random seed
        reproducibility (bool): whether to require reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if reproducibility:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


def get_tensorboard(logger):
    r""" Creates a SummaryWriter of Tensorboard that can log PyTorch models and metrics into a directory for
    visualization within the TensorBoard UI.
    For the convenience of the user, the naming rule of the SummaryWriter's log_dir is the same as the logger.

    Args:
        logger: its output filename is used to name the SummaryWriter's log_dir.
                If the filename is not available, we will name the log_dir according to the current time.

    Returns:
        SummaryWriter: it will write out events and summaries to the event file.
    """
    base_path = 'log_tensorboard'

    dir_name = None
    for handler in logger.handlers:
        if hasattr(handler, "baseFilename"):
            dir_name = os.path.basename(getattr(handler, 'baseFilename')).split('.')[0]
            break
    if dir_name is None:
        dir_name = '{}-{}'.format('model', get_local_time())

    dir_path = os.path.join(base_path, dir_name)
    writer = SummaryWriter(dir_path)
    return writer


def get_gpu_usage(device=None):
    r""" Return the reserved memory and total memory of given device in a string.
    Args:
        device: cuda.device. It is the device that the model run on.

    Returns:
        str: it contains the info about reserved memory and total memory of given device.
    """

    reserved = torch.cuda.max_memory_reserved(device) / 1024 ** 3
    total = torch.cuda.get_device_properties(device).total_memory / 1024 ** 3

    return '{:.2f} G/{:.2f} G'.format(reserved, total)
