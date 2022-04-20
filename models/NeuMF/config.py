from yacs.config import CfgNode as CN

_C = CN()

_C.SYSTEM = CN()
_C.SYSTEM.USE_GPU = True
_C.SYSTEM.DEVICE_ID = 0

_C.MODEL = CN()
_C.MODEL.NAME = 'NeuMF'
_C.MODEL.DIR = 'models/NeuMF'
_C.MODEL.SAVE_NAME = ''

# train parameters
_C.TRAIN = CN()
_C.TRAIN.DATA_TYPE = 'rt'
_C.TRAIN.DENSITY_LIST = [0.05]  # training set density, should be a list type
_C.TRAIN.BATCH_SIZE = 64
_C.TRAIN.LATENT_DIM_GMF = 8
_C.TRAIN.LATENT_DIM_MLP = 8
_C.TRAIN.NUM_EPOCHS = 20
_C.TRAIN.LAYERS = [16, 32, 16, 8]  # layes[0] is the concat of latent user vector and latent item vector

# loss function
_C.TRAIN.LOSS_FN = CN()
_C.TRAIN.LOSS_FN.TYPE = 'L1'

# optimizer
_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.TYPE = 'Adam'
_C.TRAIN.OPTIMIZER.LR = 0.0001
_C.TRAIN.OPTIMIZER.WEIGHT_DECAY = 0.00001

# pretrain
_C.TRAIN.PRETRAIN = False
_C.TRAIN.GMF_MODEL_DIR = 'models/NeuMF/pretrain/gmf_model/Density_{density}.ckpt'
_C.TRAIN.MLP_MODEL_DIR = 'models/NeuMF/pretrain/mlp_model/Density_{density}.ckpt'



def get_cfg_defaults():
    return _C.clone()
