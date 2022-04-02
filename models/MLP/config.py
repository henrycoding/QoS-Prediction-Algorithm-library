from yacs.config import CfgNode as CN

_C = CN()

_C.SYSTEM = CN()
_C.SYSTEM.USE_GPU = True
_C.SYSTEM.DEVICE_ID = 0

_C.MODEL = CN()
_C.MODEL.NAME = 'MLP'
_C.MODEL.DIR = 'models/MLP'
_C.MODEL.SAVE_NAME = ''

_C.TRAIN = CN()
# train parameters
_C.TRAIN.DATA_TYPE = 'rt'
_C.TRAIN.DENSITY_LIST = [0.05, 0.1, 0.15, 0.2]  # training set density, should be a list type
_C.TRAIN.BATCH_SIZE = 64
_C.TRAIN.LATENT_DIM = 8
_C.TRAIN.NUM_EPOCHS = 200
_C.TRAIN.LAYERS = [16, 32, 16, 8]  # layes[0] is the concat of latent user vector and latent item vector

# loss function
_C.TRAIN.LOSS_FN = CN()
_C.TRAIN.LOSS_FN.TYPE = 'L1'

# optimizer
_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.TYPE = 'Adam'
_C.TRAIN.OPTIMIZER.LR = 1e-3
_C.TRAIN.OPTIMIZER.WEIGHT_DECAY = 0.00001

# pretrain
_C.TRAIN.PRETRAIN = False
_C.TRAIN.PRETRAIN_DIR = 'models/MLP/pretrain/Density_{density}.ckpt'


def get_cfg_defaults():
    return _C.clone()
