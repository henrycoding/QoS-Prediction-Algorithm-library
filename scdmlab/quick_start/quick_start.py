import torch
import pickle
import logging
from logging import getLogger

from scdmlab.config import Config
from scdmlab.data import create_dataset, data_preparation
from scdmlab.utils import init_logger, get_model, init_seed, get_trainer


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

    for cur_density in config['density']:
        for cur_dataset_type in config['dataset_type']:
            train_data, test_data = data_preparation(config, dataset, density=cur_density,
                                                     dataset_type=cur_dataset_type)

            # model loading and initialization
            init_seed(config['seed'], config['reproducibility'])
            model = get_model(config['model'])(config, dataset).to(config['device'])
            logger.info(model)

            trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)

            trainer.fit(train_data, test_data, saved=saved, show_progress=config['show_progress'])

            test_result = trainer.evaluate(test_data, load_best_model=saved, show_progress=config['show_progress'])

    # trainer loading and initialization
    # trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)

    # model loading and initalization

    # configurations initialization
    # config = Config(model=args.model, dataset=args.dataset, dataset_type=args.dataset_type,
    #                 config_file=args.config_file)
    #
    # dataset = get_dataset(config['dataset'], config['dataset_type'])
