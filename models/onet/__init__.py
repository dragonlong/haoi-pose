import sys
TRAIN_PATH = "../../"
sys.path.insert(0, TRAIN_PATH)

from models.onet import (
    config, generation, training, models
)

__all__ = [
    config, generation, training, models
]
