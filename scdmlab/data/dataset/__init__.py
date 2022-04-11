import os
from scdmlab.config import ROOT_PATH

DATASET_DIR = os.path.join(ROOT_PATH, 'dataset', 'WSDream_1')
RT_MATRIX_DIR = os.path.join(DATASET_DIR, 'WSDream_1.rtMatrix')
TP_MATRIX_DIR = os.path.join(DATASET_DIR, 'WSDream_1.tpMatrix')
USER_DIR = os.path.join(DATASET_DIR, 'WSDream_1.user')
SERVICE_DIR = os.path.join(DATASET_DIR, 'WSDream_1.service')

from scdmlab.data.dataset.dataset import DatasetBase
