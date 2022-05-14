import os
from scdmlab.config import ROOT_PATH

WSDream_1_DIR = os.path.join(ROOT_PATH, 'dataset', 'WSDream-1')
WSDream_1_RT_MATRIX_DIR = os.path.join(WSDream_1_DIR, 'rtMatrix.txt')
WSDream_1_TP_MATRIX_DIR = os.path.join(WSDream_1_DIR, 'tpMatrix.txt')
WSDream_1_USER_DIR = os.path.join(WSDream_1_DIR, 'userlist.txt')
WSDream_1_SERVICE_DIR = os.path.join(WSDream_1_DIR, 'wslist.txt')

from scdmlab.data.dataset.abstract_dataset import AbstractDataset
from scdmlab.data.dataset.matrix_dataset import MatrixDataset
from scdmlab.data.dataset.info_dataset import InfoDataset
