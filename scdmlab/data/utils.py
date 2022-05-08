import os
import pickle
import importlib
import torch
from torch.utils.data import Dataset, DataLoader

from scdmlab.utils import ModelType, InputType


def create_dataset(config):
    dataset_module = importlib.import_module('scdmlab.data.dataset')
    if hasattr(dataset_module, config['model'] + 'Dataset'):
        dataset_class = getattr(dataset_module, config['model'] + 'Dataset')
    else:
        input_type = config['MODEL_INPUT_TYPE']
        type2class = {
            InputType.MATRIX: 'MatrixDataset',
            InputType.INFO: 'InfoDataset',
        }
        dataset_class = getattr(dataset_module, type2class[input_type])

    dataset = dataset_class(config)
    return dataset


def data_preparation(config, dataset, **kwargs):
    model_type = config['MODEL_TYPE']
    density = kwargs.get('density')
    dataset_type = kwargs.get('dataset_type')
    if model_type == ModelType.GENERAL:
        train_data, test_data = dataset.build(density, dataset_type)

        train_dataset = ToTorchDataset(train_data)
        test_dataset = ToTorchDataset(test_data)

        train_dataloader = DataLoader(train_dataset, config['train_batch_size'])
        test_dataloader = DataLoader(test_dataset, config['test_batch_size'])

        return train_dataloader, test_dataloader


class ToTorchDataset(Dataset):
    """将一个三元组转成Torch Dataset的形式
    """

    def __init__(self, triad) -> None:
        super().__init__()
        self.triad = triad
        self.user_tensor = torch.LongTensor([i[0] for i in triad])
        self.item_tensor = torch.LongTensor([i[1] for i in triad])
        self.target_tensor = torch.FloatTensor([i[2] for i in triad])

    def __len__(self):
        return len(self.triad)

    def __getitem__(self, index):
        return self.user_tensor[index], self.item_tensor[
            index], self.target_tensor[index]
