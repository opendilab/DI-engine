import torch


def get_num_params(model: torch.nn.Module) -> int:
    """
    Overview:
        Return the number of parameters in the model.
    Arguments:
        - model (:obj:`torch.nn.Module`): The model object to calculate the parameter number.
    Returns:
        - n_params (:obj:`int`): The calculated number of parameters.
    Examples:
        >>> model = torch.nn.Linear(3, 5)
        >>> num = get_num_params(model)
        >>> assert num == 15
    """
    return sum(p.numel() for p in model.parameters())
