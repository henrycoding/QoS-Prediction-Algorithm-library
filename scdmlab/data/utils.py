import os
import pickle
import importlib
import torch
from torch.utils.data import Dataset, DataLoader

from scdmlab.utils import ModelType


def create_dataset(config):
    dataset_module = importlib.import_module('scdmlab.data.dataset')  # import all modules in the dataset directory
    model_type = config['MODEL_TYPE']
    type2class = {
        ModelType.GENERAL: 'MatrixDataset',
        ModelType.CONTEXT: 'MatrixDataset',
    }
    dataset_class = getattr(dataset_module, type2class[model_type])

    default_file = os.path.join(config['checkpoint_dir'], f'{config["dataset"]}-{dataset_class.__name__}.pth')
    file = config['dataset_save_path'] or default_file
    # if os.path.exists(file):
    #     with open(file, 'rb') as f:
    #         dataset = pickle.load(f)
    #     dataset_args_unchanged = True
    #     for arg in dataset_arguments + ['seed', 'repeatable']:

    dataset = dataset_class(config)
    return dataset


# def data_preparation(config, dataset):
#     dataloaders = load_split_dataloaders(config)

def data_preparation(config, dataset, density):
    train_data, test_data = dataset.split_train_test(density)

    train_dataset = ToTorchDataset(train_data)
    test_dataset = ToTorchDataset(test_data)

    train_dataloader = DataLoader(train_dataset, config['batch_size'])
    test_dataloader = DataLoader(test_dataset, config['batch_size'])

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
