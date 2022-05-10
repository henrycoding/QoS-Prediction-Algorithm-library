from yacs.config import CfgNode as CN

_C = CN()

_C.SYSTEM = CN()
_C.SYSTEM.USE_GPU = True
_C.SYSTEM.DEVICE_ID = 0

_C.MODEL = CN()
_C.MODEL.NAME = 'LDCF'
_C.MODEL.DIR = 'models/LDCF'
_C.MODEL.SAVE_NAME = ''

# train parameters
_C.TRAIN = CN()
_C.TRAIN.DATA_TYPE = 'rt'
_C.TRAIN.DENSITY_LIST = [0.05, 0.1, 0.15, 0.2]  # training set density, should be a list type
_C.TRAIN.BATCH_SIZE = 32
_C.TRAIN.LATENT_DIM_AC = 8
_C.TRAIN.LATENT_DIM_MLP = 8
_C.TRAIN.LATENT_DIM = 8
_C.TRAIN.NUM_EPOCHS = 10
_C.TRAIN.LAYERS = [16, 64, 32, 8]  # layers[0] is the concat of latent user vector and latent item vector
_C.TRAIN.FRACTION = 0.05

# loss function
_C.TRAIN.LOSS_FN = CN()
_C.TRAIN.LOSS_FN.TYPE = 'Huber'

# optimizer
_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.TYPE = 'Adam'
_C.TRAIN.OPTIMIZER.LR = 0.0005
_C.TRAIN.OPTIMIZER.WEIGHT_DECAY = 0.00001


def get_cfg_defaults():
    return _C.clone()
