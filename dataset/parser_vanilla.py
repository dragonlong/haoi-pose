from dataset.base import SyntheticDataset as Dataset
import torch
import torch.nn as nn

def breakpoint():
    import pdb;pdb.set_trace()

class Parser():
    def __init__(self, cfg):
        self.workers       = cfg.num_workers # workers is free to decide
        self.batch_size    = cfg.TRAIN.batch_size
        self.shuffle_train =  cfg.TRAIN.shuffle_train

        self.train_dataset = Dataset(
            root_dir=cfg.root_data,
            ctgy_obj=cfg.item,
            name_dset=cfg.name_dset,
            batch_size=cfg.batch_size,
            n_max_parts=cfg.n_max_parts,
            add_noise=cfg.train_data_add_noise,
            nocs_type=cfg.nocs_type,
            parametri_type=cfg.parametri_type,
            first_n=cfg.train_first_n,
            is_debug=cfg.is_debug,
            mode='train',
            fixed_order=False,)

        self.train_sampler = None
        self.trainloader = torch.utils.data.DataLoader(self.train_dataset,
                                                     batch_size=self.batch_size,
                                                     shuffle=(self.shuffle_train and self.train_sampler is None),
                                                     num_workers=self.workers,
                                                     pin_memory=True,
                                                     drop_last=True)
        assert len(self.trainloader) > 0

        # self.trainiter = iter(self.trainloader)

        # seen instances
        self.valid_dataset = Dataset(
            root_dir=cfg.root_data,
            ctgy_obj=cfg.item,
            name_dset=cfg.name_dset,
            batch_size=cfg.batch_size,
            n_max_parts=cfg.n_max_parts,
            add_noise=cfg.val_data_add_noise,
            nocs_type=cfg.nocs_type,
            parametri_type=cfg.parametri_type,
            first_n=cfg.val_first_n,
            is_debug=cfg.is_debug,
            mode='test',
            domain='seen',
            fixed_order=True,
            )

        self.validloader = torch.utils.data.DataLoader(self.valid_dataset,
                                                       batch_size=self.batch_size,
                                                       shuffle=False,
                                                       num_workers=self.workers,
                                                       pin_memory=True,
                                                       drop_last=True)
        assert len(self.validloader) > 0
        # self.validiter = iter(self.validloader)

        # # unseen instances
        # self.test_dataset = Dataset(
        #     root_dir=cfg.root_data,
        #     ctgy_obj=cfg.item,
        #     name_dset=cfg.name_dset,
        #     batch_size=cfg.batch_size,
        #     n_max_parts=cfg.n_max_parts,
        #     add_noise=cfg.val_data_add_noise,
        #     nocs_type=cfg.nocs_type,
        #     parametri_type=cfg.parametri_type,
        #     first_n=cfg.val_first_n,
        #     is_debug=cfg.is_debug,
        #     mode='test',
        #     domain='unseen',
        #     fixed_order=True,)

        # self.testloader = torch.utils.data.DataLoader(self.test_dataset,
        #                                             batch_size=self.batch_size,
        #                                             shuffle=False,
        #                                             num_workers=self.workers,
        #                                             pin_memory=True,
        #                                             drop_last=True)
        # # assert len(self.testloader) > 0
        # self.testiter = iter(self.testloader)

    def get_train_batch(self):
        scans = self.trainiter.next()
        return scans

    def get_train_set(self):
        return self.trainloader

    def get_valid_batch(self):
        scans = self.validiter.next()
        return scans

    def get_valid_set(self):
        return self.validloader

    def get_test_batch(self):
        scans = self.testiter.next()
        return scans

    def get_test_set(self):
        return self.testloader

    def get_train_size(self):
        return len(self.trainloader)

    def get_valid_size(self):
        return len(self.validloader)

    def get_test_size(self):
        return len(self.testloader)