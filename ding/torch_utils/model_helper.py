import torch


def get_num_params(model: torch.nn.Module) -> int:
    """
    Return the number of parameters in the model.
    """
    n_params = sum(p.numel() for p in model.parameters())
    return n_params
