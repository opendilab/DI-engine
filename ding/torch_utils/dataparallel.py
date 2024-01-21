import torch
import torch.nn as nn


class DataParallel(nn.DataParallel):
    """
    Overview:
        A wrapper class for nn.DataParallel.
    Interfaces:
        ``__init__``, ``parameters``
    """

    def __init__(self, module, device_ids=None, output_device=None, dim=0):
        """
        Overview:
            Initialize the DataParallel object.
        Arguments:
            - module (:obj:`nn.Module`): The module to be parallelized.
            - device_ids (:obj:`list`): The list of GPU ids.
            - output_device (:obj:`int`): The output GPU id.
            - dim (:obj:`int`): The dimension to be parallelized.
        """
        super().__init__(module, device_ids=None, output_device=None, dim=0)
        self.module = module

    def parameters(self, recurse: bool = True):
        """
        Overview:
            Return the parameters of the module.
        Arguments:
            - recurse (:obj:`bool`): Whether to return the parameters of the submodules.
        Returns:
            - params (:obj:`generator`): The generator of the parameters.
        """
        return self.module.parameters(recurse=True)
