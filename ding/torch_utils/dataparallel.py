import torch
import torch.nn as nn


class DataParallel(nn.DataParallel):

    def __init__(self, module, device_ids=None, output_device=None, dim=0):
        super().__init__(module, device_ids=None, output_device=None, dim=0)
        self.module = module

    def parameters(self, recurse: bool = True):
        return self.module.parameters(recurse=True)
