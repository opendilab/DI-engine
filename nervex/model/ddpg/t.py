import torch
import torch.nn as nn


def hook(m, a, b):
    print('hook')


a = nn.Linear(4, 3)
a.register_backward_hook(hook)
inputs = torch.randn(2, 4)
output = a(inputs)
print('end')
