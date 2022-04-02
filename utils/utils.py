import importlib
import random
import numpy as np
import torch


def get_model(model_name: str):
    """Automatically select model class based on model name

    Args:
        model_name (str): model name

    Returns:

    """
    model_file_name = model_name.lower()
    model_module = None
    module_path = '.'.join(['models', model_file_name])
    if importlib.util.find_spec(module_path, __name__):
        model_module = importlib.import_module(module_path, __name__)

    if model_module is None:
        raise ValueError('`model_name` [{}] is not the name of an existing model.'.format(model_name))

    model_class = getattr(model_module, model_name)
    return model_class


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
