import time
import torch
from hpc_rll.origin.gae import gae, gae_data
from hpc_rll.rl_utils.gae import GAE
from testbase import mean_relative_error, times

assert torch.cuda.is_available()
use_cuda = True

T = 1024
B = 64


def gae_val():
    value = torch.randn(T + 1, B)
    reward = torch.randn(T, B)

    hpc_gae = GAE(T, B)

    if use_cuda:
        value = value.cuda()
        reward = reward.cuda()
        hpc_gae = hpc_gae.cuda()
    ori_adv = gae(gae_data(value, reward))
    hpc_adv = hpc_gae(value, reward)
    if use_cuda:
        torch.cuda.synchronize()

    mre = mean_relative_error(
        torch.flatten(ori_adv).cpu().detach().numpy(),
        torch.flatten(hpc_adv).cpu().detach().numpy()
    )
    print("gae mean_relative_error: " + str(mre))


def gae_perf():
    value = torch.randn(T + 1, B)
    reward = torch.randn(T, B)

    hpc_gae = GAE(T, B)

    if use_cuda:
        value = value.cuda()
        reward = reward.cuda()
        hpc_gae = hpc_gae.cuda()
    for i in range(times):
        t = time.time()
        adv = gae(gae_data(value, reward))
        if use_cuda:
            torch.cuda.synchronize()
        print('epoch: {}, original gae cost time: {}'.format(i, time.time() - t))
    for i in range(times):
        t = time.time()
        hpc_adv = hpc_gae(value, reward)
        if use_cuda:
            torch.cuda.synchronize()
        print('epoch: {}, hpc gae cost time: {}'.format(i, time.time() - t))


if __name__ == '__main__':
    print("target problem: T = {}, B = {}".format(T, B))
    print("================run gae validation test================")
    gae_val()
    print("================run gae performance test================")
    gae_perf()
