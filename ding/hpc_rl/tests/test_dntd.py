import time
import torch
from hpc_rll.origin.td import dist_nstep_td_error, dist_nstep_td_data
from hpc_rll.rl_utils.td import DistNStepTD
from testbase import mean_relative_error, times

assert torch.cuda.is_available()
use_cuda = True

T = 128
B = 128
N = 128
gamma = 0.95
v_min = -10.0
v_max = 10.0
n_atom = 51


def dntd_val():
    ori_dist = torch.randn(B, N, n_atom).abs()
    ori_next_n_dist = torch.randn(B, N, n_atom).abs()
    ori_action = torch.randint(0, N, size=(B, ))
    ori_next_n_action = torch.randint(0, N, size=(B, ))
    ori_reward = torch.randn(T, B)
    ori_done = torch.randn(B)
    ori_weight = torch.randn(B)

    hpc_dist = ori_dist.clone().detach()
    hpc_next_n_dist = ori_next_n_dist.clone().detach()
    hpc_action = ori_action.clone().detach()
    hpc_next_n_action = ori_next_n_action.clone().detach()
    hpc_reward = ori_reward.clone().detach()
    hpc_done = ori_done.clone().detach()
    hpc_weight = ori_weight.clone().detach()
    hpc_dntd = DistNStepTD(T, B, N, n_atom)

    if use_cuda:
        ori_dist = ori_dist.cuda()
        ori_next_n_dist = ori_next_n_dist.cuda()
        ori_action = ori_action.cuda()
        ori_next_n_action = ori_next_n_action.cuda()
        ori_reward = ori_reward.cuda()
        ori_done = ori_done.cuda()
        ori_weight = ori_weight.cuda()

        hpc_dist = hpc_dist.cuda()
        hpc_next_n_dist = hpc_next_n_dist.cuda()
        hpc_action = hpc_action.cuda()
        hpc_next_n_action = hpc_next_n_action.cuda()
        hpc_reward = hpc_reward.cuda()
        hpc_done = hpc_done.cuda()
        hpc_weight = hpc_weight.cuda()
        hpc_dntd = hpc_dntd.cuda()

    ori_dist.requires_grad_(True)
    ori_loss, ori_td_err = dist_nstep_td_error(
        dist_nstep_td_data(ori_dist, ori_next_n_dist, ori_action, ori_next_n_action, ori_reward, ori_done, ori_weight),
        gamma, v_min, v_max, n_atom, T
    )
    ori_loss = ori_loss.mean()
    ori_loss.backward()

    hpc_dist.requires_grad_(True)
    hpc_loss, hpc_td_err = hpc_dntd(
        hpc_dist, hpc_next_n_dist, hpc_action, hpc_next_n_action, hpc_reward, hpc_done, hpc_weight, gamma, v_min, v_max
    )
    hpc_loss = hpc_loss.mean()
    hpc_loss.backward()

    mre = mean_relative_error(
        torch.flatten(ori_loss).cpu().detach().numpy(),
        torch.flatten(hpc_loss).cpu().detach().numpy()
    )
    print("dntd fp mean_relative_error: " + str(mre))
    mre = mean_relative_error(
        torch.flatten(ori_td_err).cpu().detach().numpy(),
        torch.flatten(hpc_td_err).cpu().detach().numpy()
    )
    print("dntd fp td_err mean_relative_error: " + str(mre))
    mre = mean_relative_error(
        torch.flatten(ori_dist.grad).cpu().detach().numpy(),
        torch.flatten(hpc_dist.grad).cpu().detach().numpy()
    )
    print("dntd bp mean_relative_error: " + str(mre))


def dntd_perf():
    ori_dist = torch.randn(B, N, n_atom).abs()
    ori_next_n_dist = torch.randn(B, N, n_atom).abs()
    ori_action = torch.randint(0, N, size=(B, ))
    ori_next_n_action = torch.randint(0, N, size=(B, ))
    ori_reward = torch.randn(T, B)
    ori_done = torch.randn(B)
    ori_weight = torch.randn(B)

    hpc_dist = ori_dist.clone().detach()
    hpc_next_n_dist = ori_next_n_dist.clone().detach()
    hpc_action = ori_action.clone().detach()
    hpc_next_n_action = ori_next_n_action.clone().detach()
    hpc_reward = ori_reward.clone().detach()
    hpc_done = ori_done.clone().detach()
    hpc_weight = ori_weight.clone().detach()
    hpc_dntd = DistNStepTD(T, B, N, n_atom)

    if use_cuda:
        ori_dist = ori_dist.cuda()
        ori_next_n_dist = ori_next_n_dist.cuda()
        ori_action = ori_action.cuda()
        ori_next_n_action = ori_next_n_action.cuda()
        ori_reward = ori_reward.cuda()
        ori_done = ori_done.cuda()
        ori_weight = ori_weight.cuda()

        hpc_dist = hpc_dist.cuda()
        hpc_next_n_dist = hpc_next_n_dist.cuda()
        hpc_action = hpc_action.cuda()
        hpc_next_n_action = hpc_next_n_action.cuda()
        hpc_reward = hpc_reward.cuda()
        hpc_done = hpc_done.cuda()
        hpc_weight = hpc_weight.cuda()
        hpc_dntd = hpc_dntd.cuda()

    ori_dist.requires_grad_(True)
    for i in range(times):
        t = time.time()
        ori_loss, ori_td_err = dist_nstep_td_error(
            dist_nstep_td_data(
                ori_dist, ori_next_n_dist, ori_action, ori_next_n_action, ori_reward, ori_done, ori_weight
            ), gamma, v_min, v_max, n_atom, T
        )
        ori_loss = ori_loss.mean()
        ori_loss.backward()
        if use_cuda:
            torch.cuda.synchronize()
        print('epoch: {}, origin dntd cost time: {}'.format(i, time.time() - t))

    hpc_dist.requires_grad_(True)
    for i in range(times):
        t = time.time()
        hpc_loss, hpc_td_err = hpc_dntd(
            hpc_dist, hpc_next_n_dist, hpc_action, hpc_next_n_action, hpc_reward, hpc_done, hpc_weight, gamma, v_min,
            v_max
        )
        hpc_loss = hpc_loss.mean()
        hpc_loss.backward()
        if use_cuda:
            torch.cuda.synchronize()
        print('epoch: {}, hpc dntd cost time: {}'.format(i, time.time() - t))


if __name__ == '__main__':
    print(
        "target problem: T = {}, B = {}, N = {}, gamma = {}, v_min = {}, v_max = {}, n_atom = {}".format(
            T, B, N, gamma, v_min, v_max, n_atom
        )
    )
    print("================run dntd validation test================")
    dntd_val()
    print("================run dntd performance test================")
    dntd_perf()
