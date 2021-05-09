import sys
TRAIN_PATH = "../"
sys.path.insert(0, TRAIN_PATH)

from models.agent_ae import PointVAEAgent, PointAEPoseAgent
from models.agent_gan import MainAgent

def get_agent(config):
    if config.module == 'ae':
        return PointAEPoseAgent(config)
    elif config.module == 'vae':
        return PointVAEAgent(config)
    elif config.module == 'gan':
        return MainAgent(config)
    else:
        raise ValueError
