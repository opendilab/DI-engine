import torch


def is_differentiable(loss, model):
    assert isinstance(loss, torch.Tensor)
    assert isinstance(model, torch.nn.Module)
    for p in model.parameters():
        assert p.grad is None
    loss.backward()
    for k, p in model.named_parameters():
        assert isinstance(p.grad, torch.Tensor), k
