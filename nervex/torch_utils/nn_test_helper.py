from typing import Union, List
import torch


def is_differentiable(loss: torch.Tensor, model: Union[torch.Tensor, List[torch.Tensor]]) -> None:
    """
    Overview:
        Judge whether the model/models are differentiable. First check whether module's grad is None,
        then do loss's back propagation, finally check whether module's grad are torch.Tensor.
    Arguments:
        - loss (:obj:`torch.Tensor`): loss tensor of the model
        - model (:obj:`Union[torch.Tensor, List[torch.Tensor]]`): model or models to be checked
    """
    assert isinstance(loss, torch.Tensor)
    if isinstance(model, list):
        for m in model:
            assert isinstance(m, torch.nn.Module)
            for k, p in m.named_parameters():
                assert p.grad is None, k
    elif isinstance(model, torch.nn.Module):
        for k, p in model.named_parameters():
            assert p.grad is None, k
    else:
        raise TypeError('model must be list or nn.Module')

    loss.backward()

    if isinstance(model, list):
        for m in model:
            for k, p in m.named_parameters():
                assert isinstance(p.grad, torch.Tensor), k
    elif isinstance(model, torch.nn.Module):
        for k, p in model.named_parameters():
            assert isinstance(p.grad, torch.Tensor), k

