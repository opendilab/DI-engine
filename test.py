import torch

cur_seq = 6
prev_seq = 6
b = torch.ones(cur_seq, cur_seq+prev_seq)
a = torch.triu(b, diagonal=1+prev_seq)
print(a)