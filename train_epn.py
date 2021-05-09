import os
from time import time
import hydra
from hydra import utils
import logging
from omegaconf import DictConfig, ListConfig, OmegaConf
import torch

from global_info import global_info
from models.trainer_modelnetDirectRotation import Trainer as TrainerR
from models.trainer_modelnetRotation import Trainer

def bp():
    import pdb;pdb.set_trace()

# @hydra.main(config_path="config/models/epn.yaml")
@hydra.main(config_path="config/completion.yaml")
def main(cfg):
    OmegaConf.set_struct(cfg, False)  # This allows getattr and hasattr methods to function correctly
    #>>>>>>>>>>>>>>>>> setting <<<<<<<<<<<<<<<<<<< #
    os.chdir(hydra.utils.get_original_cwd())
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    is_cuda = (torch.cuda.is_available() and not cfg.no_cuda)
    device = torch.device("cuda" if is_cuda else "cpu")

    t0 = time()
    infos       = global_info()
    data_infos  = infos.datasets[cfg.item]
    cfg.root_data   = infos.second_path + '/data'
    cfg.log_dir     = infos.second_path + cfg.log_dir
    if not os.path.isdir(cfg.log_dir):
        os.makedirs(cfg.log_dir)
        os.makedirs(cfg.log_dir + '/checkpoints'
        )

    if cfg.eval:
        cfg.resume_path = 'airplane_net_Iterbest.pth' # airplane_net_Iter110000.pth -
    trainer = TrainerR(cfg)

    if cfg.eval:
        trainer.eval()
    else:
        trainer.train()

if __name__ == '__main__':
    main()
