import sys
TRAIN_PATH = "../../"
sys.path.insert(0, TRAIN_PATH)

from models.spconv.base_so3conv import *
from models.spconv import inv_so3net
from models.spconv import cls_so3net_pn
from models.spconv import reg_so3net
from models.spconv import dir_so3net
from models.spconv import enc_so3net
