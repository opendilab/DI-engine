import time
import torch
from hpc_rll.origin.upgo import upgo_loss
from hpc_rll.rl_utils.upgo import UPGO
from testbase import mean_relative_error, times

assert torch.cuda.is_available()
use_cuda = True

T = 256
B = 256
N = 256


def upgo_val():
    ori_target_output = torch.randn(T, B, N)
    ori_rhos = torch.randn(T, B)
    ori_action = torch.randint(
        0, N, size=(
            T,
            B,
        )
    )
    ori_rewards = torch.randn(T, B)
    ori_bootstrap_values = torch.randn(T + 1, B)

    hpc_target_output = ori_target_output.clone().detach()
    hpc_rhos = ori_rhos.clone().detach()
    hpc_action = ori_action.clone().detach()
    hpc_rewards = ori_rewards.clone().detach()
    hpc_bootstrap_values = ori_bootstrap_values.clone().detach()
    hpc_upgo = UPGO(T, B, N)

    if use_cuda:
        ori_target_output = ori_target_output.cuda()
        ori_rhos = ori_rhos.cuda()
        ori_action = ori_action.cuda()
        ori_rewards = ori_rewards.cuda()
        ori_bootstrap_values = ori_bootstrap_values.cuda()

        hpc_target_output = hpc_target_output.cuda()
        hpc_rhos = hpc_rhos.cuda()
        hpc_action = hpc_action.cuda()
        hpc_rewards = hpc_rewards.cuda()
        hpc_bootstrap_values = hpc_bootstrap_values.cuda()
        hpc_upgo = hpc_upgo.cuda()

    ori_target_output.requires_grad_(True)
    ori_loss = upgo_loss(ori_target_output, ori_rhos, ori_action, ori_rewards, ori_bootstrap_values)
    ori_loss = ori_loss.mean()
    ori_loss.backward()
    if use_cuda:
        torch.cuda.synchronize()

    hpc_target_output.requires_grad_(True)
    hpc_loss = hpc_upgo(hpc_target_output, hpc_rhos, hpc_action, hpc_rewards, hpc_bootstrap_values)
    hpc_loss = hpc_loss.mean()
    hpc_loss.backward()
    if use_cuda:
        torch.cuda.synchronize()

    mre = mean_relative_error(
        torch.flatten(ori_loss).cpu().detach().numpy(),
        torch.flatten(hpc_loss).cpu().detach().numpy()
    )
    print("upgo fp mean_relative_error: " + str(mre))
    mre = mean_relative_error(
        torch.flatten(ori_target_output.grad).cpu().detach().numpy(),
        torch.flatten(hpc_target_output.grad).cpu().detach().numpy()
    )
    print("upgo bp mean_relative_error: " + str(mre))


def upgo_perf():
    ori_target_output = torch.randn(T, B, N)
    ori_rhos = torch.randn(T, B)
    ori_action = torch.randint(
        0, N, size=(
            T,
            B,
        )
    )
    ori_rewards = torch.randn(T, B)
    ori_bootstrap_values = torch.randn(T + 1, B)

    hpc_target_output = ori_target_output.clone().detach()
    hpc_rhos = ori_rhos.clone().detach()
    hpc_action = ori_action.clone().detach()
    hpc_rewards = ori_rewards.clone().detach()
    hpc_bootstrap_values = ori_bootstrap_values.clone().detach()
    hpc_upgo = UPGO(T, B, N)

    if use_cuda:
        ori_target_output = ori_target_output.cuda()
        ori_rhos = ori_rhos.cuda()
        ori_action = ori_action.cuda()
        ori_rewards = ori_rewards.cuda()
        ori_bootstrap_values = ori_bootstrap_values.cuda()

        hpc_target_output = hpc_target_output.cuda()
        hpc_rhos = hpc_rhos.cuda()
        hpc_action = hpc_action.cuda()
        hpc_rewards = hpc_rewards.cuda()
        hpc_bootstrap_values = hpc_bootstrap_values.cuda()
        hpc_upgo = hpc_upgo.cuda()

    ori_target_output.requires_grad_(True)
    for i in range(times):
        t = time.time()
        ori_loss = upgo_loss(ori_target_output, ori_rhos, ori_action, ori_rewards, ori_bootstrap_values)
        ori_loss = ori_loss.mean()
        ori_loss.backward()
        if use_cuda:
            torch.cuda.synchronize()
        print('epoch: {}, original upgo cost time: {}'.format(i, time.time() - t))

    hpc_target_output.requires_grad_(True)
    for i in range(times):
        t = time.time()
        hpc_loss = hpc_upgo(hpc_target_output, hpc_rhos, hpc_action, hpc_rewards, hpc_bootstrap_values)
        hpc_loss = hpc_loss.mean()
        hpc_loss.backward()
        if use_cuda:
            torch.cuda.synchronize()
        print('epoch: {}, hpc upgo cost time: {}'.format(i, time.time() - t))


if __name__ == '__main__':
    print("target problem: T = {}, B = {}, N = {}".format(T, B, N))
    print("================run upgo validation test================")
    upgo_val()
    print("================run upgo performance test================")
    upgo_perf()
