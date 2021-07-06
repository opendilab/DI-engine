import time
import torch
from hpc_rll.origin.td import td_lambda_error, td_lambda_data
from hpc_rll.rl_utils.td import TDLambda
from testbase import mean_relative_error, times

assert torch.cuda.is_available()
use_cuda = True

T = 1024
B = 64


def td_val():
    ori_value = torch.randn(T + 1, B)
    ori_reward = torch.randn(T, B)
    ori_weight = torch.randn(T, B)

    hpc_value = ori_value.clone().detach()
    hpc_reward = ori_reward.clone().detach()
    hpc_weight = ori_weight.clone().detach()
    hpc_td = TDLambda(T, B)

    if use_cuda:
        ori_value = ori_value.cuda()
        ori_reward = ori_reward.cuda()
        ori_weight = ori_weight.cuda()

        hpc_value = hpc_value.cuda()
        hpc_reward = hpc_reward.cuda()
        hpc_weight = hpc_weight.cuda()
        hpc_td = hpc_td.cuda()

    ori_value.requires_grad_(True)
    ori_loss = td_lambda_error(td_lambda_data(ori_value, ori_reward, ori_weight))
    ori_loss = ori_loss.mean()
    ori_loss.backward()
    if use_cuda:
        torch.cuda.synchronize()

    hpc_value.requires_grad_(True)
    hpc_loss = hpc_td(hpc_value, hpc_reward, hpc_weight)
    hpc_loss = hpc_loss.mean()
    hpc_loss.backward()
    if use_cuda:
        torch.cuda.synchronize()

    mre = mean_relative_error(
        torch.flatten(ori_loss).cpu().detach().numpy(),
        torch.flatten(hpc_loss).cpu().detach().numpy()
    )
    print("td fp mean_relative_error: " + str(mre))
    mre = mean_relative_error(
        torch.flatten(ori_value.grad).cpu().detach().numpy(),
        torch.flatten(hpc_value.grad).cpu().detach().numpy()
    )
    print("td bp mean_relative_error: " + str(mre))


def td_perf():
    ori_value = torch.randn(T + 1, B)
    ori_reward = torch.randn(T, B)
    ori_weight = torch.randn(T, B)

    hpc_value = ori_value.clone().detach()
    hpc_reward = ori_reward.clone().detach()
    hpc_weight = ori_weight.clone().detach()
    hpc_td = TDLambda(T, B)

    if use_cuda:
        ori_value = ori_value.cuda()
        ori_reward = ori_reward.cuda()
        ori_weight = ori_weight.cuda()

        hpc_value = hpc_value.cuda()
        hpc_reward = hpc_reward.cuda()
        hpc_weight = hpc_weight.cuda()
        hpc_td = hpc_td.cuda()

    ori_value.requires_grad_(True)
    for i in range(times):
        t = time.time()
        ori_loss = td_lambda_error(td_lambda_data(ori_value, ori_reward, ori_weight))
        ori_loss = ori_loss.mean()
        ori_loss.backward()
        if use_cuda:
            torch.cuda.synchronize()
        print('epoch: {}, original td cost time: {}'.format(i, time.time() - t))

    hpc_value.requires_grad_(True)
    for i in range(times):
        t = time.time()
        hpc_loss = hpc_td(hpc_value, hpc_reward, hpc_weight)
        hpc_loss = hpc_loss.mean()
        hpc_loss.backward()
        if use_cuda:
            torch.cuda.synchronize()
        print('epoch: {}, hpc td cost time: {}'.format(i, time.time() - t))


if __name__ == '__main__':
    print("target problem: T = {}, B = {}".format(T, B))
    print("================run td validation test================")
    td_val()
    print("================run td performance test================")
    td_perf()
