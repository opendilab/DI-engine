import torch


def is_differentiable(loss, model):
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

