import os
from moftransformer import __root_dir__

DEFAULT_PRETRAIN_MODEL_PATH = os.path.join(__root_dir__, 'database/pretrained_model.ckpt')
DEFAULT_FINETUNED_MODEL_PATH = os.path.join(__root_dir__, 'database/finetuned/')
DEFAULT_COREMOF_PATH = os.path.join(__root_dir__, 'database/coremof/')
DEFAULT_QMOF_PATH = os.path.join(__root_dir__, 'database/qmof/')
DEFAULT_HMOF_PATH = os.path.join(__root_dir__, 'database/hmof/')