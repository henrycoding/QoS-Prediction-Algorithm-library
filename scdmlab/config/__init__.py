import os

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
BASE_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

from scdmlab.config.configurator import Config
