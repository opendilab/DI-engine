import torch


def get_num_params(model: torch.nn.Module) -> int:
    """
    Overview:
        Return the number of parameters in the model.
    Arguments:
        - model (:obj:`torch.nn.Module`): The model object to calculate the parameter number.
    Returns:
        - n_params (:obj:`int`): The calculated number of parameters.
    """
    n_params = sum(p.numel() for p in model.parameters())
    return n_params
