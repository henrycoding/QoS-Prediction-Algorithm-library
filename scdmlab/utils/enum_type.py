from enum import Enum


class ModelType(Enum):
    TRADITIONAL = 1
    GENERAL = 2
    MULTITASK = 3


class InputType(Enum):
    MATRIX = 1
    INFO = 2
