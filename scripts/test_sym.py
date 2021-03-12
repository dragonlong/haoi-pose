import sys
import os
import numpy as np
import __init__
from common.debugger import *
from common.d3_utils import rotate_about_axis
from global_info import global_info


# global variables
infos           = global_info()
sym_type        = infos.sym_type
all_rmats = [np.eye(3)]
target_category = 'can'
for key, M in sym_type[target_category].items():
    next_rmats = []
    for k in range(M):
        rmat = rotate_about_axis(2 * np.pi * k / M, axis=key)
        for old_rmat in all_rmats:
            next_rmats.append(np.matmul(rmat, old_rmat))
    all_rmats = next_rmats

print(len(all_rmats))
