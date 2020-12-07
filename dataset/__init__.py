# print('we are using dataset init!!!')
import sys
TRAIN_PATH = "../"
DEPLOY_PATH = "../deploy"
sys.path.insert(0, TRAIN_PATH)

from dataset.core import (
    Shapes3dDataset, collate_remove_none, worker_init_fn
)
from dataset.fields import (
    IndexField, PointsField,
    VoxelsField, PatchPointsField, PointCloudField, PatchPointCloudField, PartialPointCloudField,
)
from dataset.transforms import (
    PointcloudNoise, SubsamplePointcloud,
    SubsamplePoints,
)
__all__ = [
    # Core
    Shapes3dDataset,
    collate_remove_none,
    worker_init_fn,
    # Fields
    IndexField,
    PointsField,
    VoxelsField,
    PointCloudField,
    PartialPointCloudField,
    PatchPointCloudField,
    PatchPointsField,
    # Transforms
    PointcloudNoise,
    SubsamplePointcloud,
    SubsamplePoints,
]
