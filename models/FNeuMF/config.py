from yacs.config import CfgNode as CN

_C = CN()

_C.SYSTEM = CN()
_C.SYSTEM.USE_GPU = True
_C.SYSTEM.DEVICE_ID = 0

_C.MODEL = CN()
_C.MODEL.NAME = 'NeuMF'
_C.MODEL.DIR = 'models/NeuMF'
_C.MODEL.LOAD_PATH = 'output/NeuMF/2022-04-07_13-52-20/saved_model'
_C.MODEL.SAVE_NAME = ''

# train parameters
_C.TRAIN = CN()
_C.TRAIN.DATA_TYPE = 'rt'
_C.TRAIN.DENSITY_LIST = [0.05, 0.1, 0.15, 0.2]  # training set density, should be a list type
_C.TRAIN.BATCH_SIZE = 2048
_C.TRAIN.LATENT_DIM_GMF = 8
_C.TRAIN.LATENT_DIM_MLP = 8
_C.TRAIN.LATENT_DIM = 8
_C.TRAIN.NUM_EPOCHS = 50
_C.TRAIN.LAYERS = [64, 32, 8]  # layers[0] is the concat of latent user vector and latent item vector
_C.TRAIN.FRACTION = 0.1

# loss function
_C.TRAIN.LOSS_FN = CN()
_C.TRAIN.LOSS_FN.TYPE = 'L1'

# optimizer
_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.TYPE = 'Adam'
_C.TRAIN.OPTIMIZER.LR = 0.0005
_C.TRAIN.OPTIMIZER.WEIGHT_DECAY = 0.00001

# pretrain
_C.TRAIN.PRETRAIN = False
_C.TRAIN.GMF_MODEL_DIR = 'models/NeuMF/pretrain/gmf_model/Density_{density}.ckpt'
_C.TRAIN.MLP_MODEL_DIR = 'models/NeuMF/pretrain/mlp_model/Density_{density}.ckpt'


def get_cfg_defaults():
    return _C.clone()
