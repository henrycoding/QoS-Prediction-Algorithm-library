import torch
import pickle
import logging
from logging import getLogger

from scdmlab.config import Config
from scdmlab.data import create_dataset, data_preparation
from scdmlab.utils import init_logger, get_model, init_seed


def run_scdmlab(model=None, dataset=None, config_file_list=None, config_dict=None, saved=True):
    # configurations initialization
    config = Config(model=model, dataset=dataset, config_file_list=config_file_list,
                    config_dict=config_dict)
    init_seed(config['seed'], config['reproducibility'])

    # logger initialization
    init_logger(config)
    logger = getLogger()
    logger.info(config)

    # dataset create
    dataset = create_dataset(config)

    # TODO 检查是否有这两个属性，以及将属性变为list类型
    for density in config['density']:
        for dataset_type in config['dataset_type']:
            config['current_density'] = density
            config['current_dataset_type'] = dataset_type
            train_data, test_data = data_preparation(config, dataset)

    # model loading and initialization
    init_seed(config['seed'], config['reproducibility'])
    model = get_model(config['model'])(config)

    # trainer loading and initialization
    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)

    # model loading and initalization

    # configurations initialization
    # config = Config(model=args.model, dataset=args.dataset, dataset_type=args.dataset_type,
    #                 config_file=args.config_file)
    #
    # dataset = get_dataset(config['dataset'], config['dataset_type'])
