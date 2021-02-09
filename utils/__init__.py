"""
contains:
1. external cuda/cpp libraries: sampling, mesh, voxelize;
2. post-processing;
3. evaluation metrics;(some also in models/losses.py)
    ioueval.py
    metric_util.py
"""
import sys
TRAIN_PATH = "../"
sys.path.insert(0, TRAIN_PATH)
