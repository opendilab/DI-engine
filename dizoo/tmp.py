# import torch.nn as nn
# import torch
#
# from ding.torch_utils import Adam, to_device
#
# # With Learnable Parameters
# m = nn.BatchNorm2d(100)
# for k, v in m.named_parameters():
#     print(k, v)
# optimizer = Adam(m.parameters(), lr=1e-3)
#
# input = torch.randn(20, 100, 35, 45)
# output = m(input)
# loss = output.sum()
#
# # update
# optimizer.zero_grad()
# loss.backward()
# for k, v in m.named_parameters():
#     print(k, v.grad)
#
# optimizer.step()

import torch
x = torch.randn((1,4),dtype=torch.float32,requires_grad=True)
y = x ** 2
z = x * 4
output1 = z.mean()
output2 = z.sum()
print(x.grad)
output1.backward(retain_graph=True)   # 这里参数表明保留backward后的中间参数。
print(x.grad.data)

output2.backward()
print(x.grad.data)

