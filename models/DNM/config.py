from yacs.config import CfgNode as CN

_C = CN()
_C.model_name = 'DNM'
_C.model_dir = 'models/DNM'
_C.use_gpu = True
_C.device_id = 0

_C.density = 0.05
_C.embedding_dim = 8
_C.perception_layers = [16, 32]
_C.task_specific_layers_rt = [32, 16, 8]
_C.task_specific_layers_tp = [32, 16, 8]
_C.user_enable_columns = ["[User ID]"]
# _C.user_enable_columns = ["[User ID]", "[Country]", "[AS]"]
_C.service_enable_columns = ["[Service ID]"]
# _C.service_enable_columns = ["[Service ID]", "[Country]", "[AS]"]

_C.batch_size = 128
_C.num_epochs = 200
_C.loss_fn = 'L1'
_C.optimizer = 'Adam'
_C.lr = 0.001
_C.weight_decay = 0.

_C.SYSTEM = CN()
_C.SYSTEM.USE_GPU = True
_C.SYSTEM.DEVICE_ID = 0

_C.MODEL = CN()
_C.MODEL.NAME = 'DNM'
_C.MODEL.DIR = 'models/DNM'
_C.MODEL.SAVE_NAME = ''

# train parameters
_C.TRAIN = CN()
_C.TRAIN.DATA_TYPE = 'rt'
_C.TRAIN.DENSITY_LIST = [0.05]  # training set density, should be a list type
_C.TRAIN.BATCH_SIZE = 64
_C.TRAIN.LATENT_DIM_GMF = 8
_C.TRAIN.LATENT_DIM_MLP = 8
_C.TRAIN.NUM_EPOCHS = 200

_C.TRAIN.LAYERS = [16, 32, 16, 8]  # layes[0] is the concat of latent user vector and latent item vector

# loss function
_C.TRAIN.LOSS_FN = CN()
_C.TRAIN.LOSS_FN.TYPE = 'L1'

# optimizer
_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.TYPE = 'Adam'
_C.TRAIN.OPTIMIZER.LR = 0.001
_C.TRAIN.OPTIMIZER.WEIGHT_DECAY = 0

# pretrain
_C.TRAIN.PRETRAIN = False
_C.TRAIN.GMF_MODEL_DIR = 'models/NeuMF/pretrain/gmf_model/Density_{density}.ckpt'
_C.TRAIN.MLP_MODEL_DIR = 'models/NeuMF/pretrain/mlp_model/Density_{density}.ckpt'


def get_cfg_defaults():
    return _C.clone()
