from typing import Union, List
import torch


def is_differentiable(
        loss: torch.Tensor, model: Union[torch.nn.Module, List[torch.nn.Module]], print_instead: bool = False
) -> None:
    """
    Overview:
        Judge whether the model/models are differentiable. First check whether module's grad is None,
        then do loss's back propagation, finally check whether module's grad are torch.Tensor.
    Arguments:
        - loss (:obj:`torch.Tensor`): loss tensor of the model
        - model (:obj:`Union[torch.nn.Module, List[torch.nn.Module]]`): model or models to be checked
        - print_instead (:obj:`bool`): Whether to print module's final grad result, \
            instead of asserting. Default set to ``False``.
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
                if print_instead:
                    if not isinstance(p.grad, torch.Tensor):
                        print(k, "grad is:", p.grad)
                else:
                    assert isinstance(p.grad, torch.Tensor), k
    elif isinstance(model, torch.nn.Module):
        for k, p in model.named_parameters():
            if print_instead:
                if not isinstance(p.grad, torch.Tensor):
                    print(k, "grad is:", p.grad)
            else:
                assert isinstance(p.grad, torch.Tensor), k
