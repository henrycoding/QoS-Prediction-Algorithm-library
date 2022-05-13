import re
import os
import sys
from enum import Enum

import yaml
import torch
from logging import getLogger
from scdmlab.config import BASE_PATH, ROOT_PATH
from scdmlab.utils import get_model, general_arguments, training_arguments, evaluation_arguments, set_color, ModelType, \
    InputType


class Config(object):
    def __init__(self, model=None, dataset=None, config_file_list=None, config_dict=None):
        """
        Args:
            model (str/AbstractModel): the model name or the model class, default is None, if it is None, config will search the parameter 'model' from the external input as the model name or model class.
            dataset (str): the dataset name, default is None, if it is None, config will search the parameter 'dataset'
            config_file_list (list of str): the external config file, it allows multiple config files, default is None.
            config_dict (dict): the external parameter dictionaries, default is None.
        """
        self._init_parameters_category()
        self._merge_external_config_dict(config_file_list, config_dict)
        self.model, self.model_class, self.dataset = self._get_model_and_dataset(model, dataset)
        self._load_internal_config_dict(self.model, self.model_class, self.dataset)
        self.final_config_dict = self._get_final_config_dict()
        self._set_default_parameters()
        self._init_device()

    def _init_parameters_category(self):
        """Initialize the keys of the basic parameters of the parameter dictionary
        """
        self.parameters = dict()
        self.parameters['General'] = general_arguments
        self.parameters['Training'] = training_arguments
        self.parameters['Evaluation'] = evaluation_arguments

    def _convert_config_dict(self, config_dict):
        """This function convert the str parameters to their original type.
        """
        for key in config_dict:
            param = config_dict[key]
            if not isinstance(param, str):
                continue
            try:
                value = eval(param)
                if value is not None and not isinstance(value, (str, int, float, list, tuple, dict, bool, Enum)):
                    value = param
            except (NameError, SyntaxError, TypeError):
                if isinstance(param, str):
                    if param.lower() == "true":
                        value = True
                    elif param.lower() == "false":
                        value = False
                    else:
                        value = param
                else:
                    value = param
            config_dict[key] = value
        return config_dict

    def _load_config_files(self, file_list):
        """ Read parameters from configurator files
        """
        file_config_dict = dict()
        if file_list:
            for file in file_list:
                with open(file, 'r', encoding='utf-8') as f:
                    file_config_dict.update(yaml.safe_load(f.read()))
        return file_config_dict

    def _load_variable_config_dict(self, config_dict):
        """ Read parameters from variable dictionary
        """
        return self._convert_config_dict(config_dict) if config_dict else dict()

    def _load_cmd_line(self):
        """ Read parameters from command line
        """
        cmd_config_dict = dict()
        unrecognized_args = []
        if "ipykernel_launcher" not in sys.argv[0]:
            for arg in sys.argv[1:]:
                if not arg.startswith("--") or len(arg[2:].split("=")) != 2:
                    unrecognized_args.append(arg)
                    continue
                cmd_arg_name, cmd_arg_value = arg[2:].split("=")
                if cmd_arg_name in cmd_config_dict and cmd_arg_value != cmd_config_dict[cmd_arg_name]:
                    raise SyntaxError(f"There are duplicate commend arg '{arg}' with different value.")
                else:
                    cmd_config_dict[cmd_arg_name] = cmd_arg_value
        if len(unrecognized_args) > 0:
            logger = getLogger()
            logger.warning(f'command line args [{unrecognized_args}] will not be used')
        cmd_config_dict = self._convert_config_dict(cmd_config_dict)
        return cmd_config_dict

    def _merge_external_config_dict(self, config_file_list, config_dict):
        """Merge external configuration dictionaries
        """
        file_config_dict = self._load_config_files(config_file_list)
        variable_config_dict = self._load_variable_config_dict(config_dict)
        cmd_config_dict = self._load_cmd_line()

        external_config_dict = dict()
        external_config_dict.update(file_config_dict)
        external_config_dict.update(variable_config_dict)
        external_config_dict.update(cmd_config_dict)

        self.external_config_dict = external_config_dict

    def _get_model_and_dataset(self, model, dataset):
        if model is None:
            try:
                model = self.external_config_dict['model']
            except KeyError:
                raise KeyError(
                    'model need to be specified in at least one of the these ways: [model variable, config file, config dict, command line]'
                )
        if not isinstance(model, str):
            final_model_class = model
            final_model = model.__name__
        else:
            final_model = model
            final_model_class = get_model(final_model)

        if dataset is None:
            try:
                final_dataset = self.external_config_dict['dataset']
            except KeyError:
                raise KeyError(
                    'dataset need to be specified in at least one of the these ways: [dataset variable, config file, config dict, command line]'
                )
        else:
            final_dataset = dataset

        return final_model, final_model_class, final_dataset

    def _update_internal_config_dict(self, file):
        with open(file, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f.read())
            if config_dict is not None:
                self.internal_config_dict.update(config_dict)
        return config_dict

    def _load_internal_config_dict(self, model, model_class, dataset):
        overall_init_file = os.path.join(BASE_PATH, 'properties', 'overall.yaml')
        model_init_file = os.path.join(BASE_PATH, 'properties', 'model', str(model) + '.yaml')

        self.internal_config_dict = dict()
        for file in [overall_init_file, model_init_file]:
            if os.path.isfile(file):
                self._update_internal_config_dict(file)
        self.internal_config_dict['MODEL_TYPE'] = model_class.model_type

    def _get_final_config_dict(self):
        final_config_dict = dict()
        final_config_dict.update(self.internal_config_dict)
        final_config_dict.update(self.external_config_dict)
        return final_config_dict

    def _set_default_parameters(self):
        self.final_config_dict['dataset'] = self.dataset
        self.final_config_dict['model'] = self.model
        if self.dataset == 'WSDream-1':
            self.final_config_dict['data_path'] = os.path.join(ROOT_PATH, 'dataset', self.dataset)

        if hasattr(self.model_class, 'input_type'):
            self.final_config_dict['INPUT_TYPE'] = self.model_class.input_type
        else:
            raise ValueError("Model should has attr 'input_type'")

        density = self.final_config_dict['density']
        if isinstance(density, str):
            self.final_config_dict['density'] = [density]

        dataset_type = self.final_config_dict['dataset_type']
        if isinstance(dataset_type, str):
            self.final_config_dict['dataset_type'] = [dataset_type]

        # TODO metrics类实现
        metrics = self.final_config_dict['metrics']
        if isinstance(metrics, str):
            self.final_config_dict['metrics'] = [metrics]

    def _init_device(self):
        use_gpu = self.final_config_dict['use_gpu']
        if use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.final_config_dict['gpu_id'])
        self.final_config_dict['device'] = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")

    def __setitem__(self, key, value):
        if not isinstance(key, str):
            raise TypeError("index must be a str.")
        self.final_config_dict[key] = value

    def __getattr__(self, item):
        if 'final_config_dict' not in self.__dict__:
            raise AttributeError(f"'Config' object has no attribute 'final_config_dict'")
        if item in self.final_config_dict:
            return self.final_config_dict[item]
        raise AttributeError(f"'Config' object has no attribute '{item}'")

    def __getitem__(self, item):
        if item in self.final_config_dict:
            return self.final_config_dict[item]
        else:
            return None

    def __contains__(self, key):
        if not isinstance(key, str):
            raise TypeError("index must be a str.")
        return key in self.final_config_dict

    def __str__(self):
        args_info = '\n'
        for category in self.parameters:
            args_info += set_color(category + ' Hyper Parameters:\n', 'pink')
            args_info += '\n'.join([(set_color("{}", 'cyan') + " =" + set_color(" {}", 'yellow')).format(arg, value)
                                    for arg, value in self.final_config_dict.items()
                                    if arg in self.parameters[category]])
            args_info += '\n\n'

        args_info += set_color('Other Hyper Parameters: \n', 'pink')
        args_info += '\n'.join([
            (set_color("{}", 'cyan') + " = " + set_color("{}", 'yellow')).format(arg, value)
            for arg, value in self.final_config_dict.items()
            if arg not in {
                _ for args in self.parameters.values() for _ in args
            }.union({'model', 'dataset', 'config_files'})
        ])
        args_info += '\n\n'
        return args_info

    def __repr__(self):
        return self.__str__()
