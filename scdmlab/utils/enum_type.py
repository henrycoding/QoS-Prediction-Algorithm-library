from enum import Enum


class ModelType(Enum):
    TRADITIONAL = 1
    GENERAL = 2
    CONTEXT = 3


class InputType(Enum):
    POINTWISE = 1
