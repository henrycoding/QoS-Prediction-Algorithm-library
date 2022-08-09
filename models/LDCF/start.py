from models.LDCF.config import get_cfg_defaults
from models.LDCF.model import LDCFModel
from utils.model_util import ModelTest

cfg = get_cfg_defaults()
test = ModelTest(LDCFModel, cfg)
test.run(is_fed=True)
