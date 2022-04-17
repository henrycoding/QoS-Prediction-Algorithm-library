import torch
import pickle
import logging
from logging import getLogger

from scdmlab.config import Config
from scdmlab.data import create_dataset, data_preparation
from scdmlab.utils import init_logger, get_model, init_seed


def run_scdmlab(model=None, dataset=None, dataset_type=None, config_file_list=None, config_dict=None, saved=True):
    # configurations initialization
    config = Config(model=model, dataset=dataset, dataset_type=dataset_type, config_file_list=config_file_list,
                    config_dict=config_dict)
    init_seed(config['seed'], config['reproducibility'])

    # logger initialization
    init_logger(config)
    logger = getLogger()
    logger.info(config)

    # for type_ in config['dataset_type']:
    #     # dataset preparing
    #     dataset = create_dataset(config)
    #     for density in config['density']:
    #         # dataset splitting
    #         train_dataloader, test_dataloader = data_preparation(config, dataset, density)
    #
    #         # model loading and initialization
    #         model = get_model(config['model'])(config, train_dataloader).to(config['device'])

            # trainer loading and initialization
            # trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)

    # model loading and initalization

    # configurations initialization
    # config = Config(model=args.model, dataset=args.dataset, dataset_type=args.dataset_type,
    #                 config_file=args.config_file)
    #
    # dataset = get_dataset(config['dataset'], config['dataset_type'])
