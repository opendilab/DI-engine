import time
import torch
from hpc_rll.origin.td import q_nstep_td_error, q_nstep_td_data
from hpc_rll.rl_utils.td import QNStepTD
from testbase import mean_relative_error, times

assert torch.cuda.is_available()
use_cuda = True

T = 1024
B = 64
N = 64
gamma = 0.95


def qntd_val():
    ori_q = torch.randn(B, N)
    ori_next_n_q = torch.randn(B, N)
    ori_action = torch.randint(0, N, size=(B, ))
    ori_next_n_action = torch.randint(0, N, size=(B, ))
    ori_reward = torch.randn(T, B)
    ori_done = torch.randn(B)
    ori_weight = torch.randn(B)

    hpc_q = ori_q.clone().detach()
    hpc_next_n_q = ori_next_n_q.clone().detach()
    hpc_action = ori_action.clone().detach()
    hpc_next_n_action = ori_next_n_action.clone().detach()
    hpc_reward = ori_reward.clone().detach()
    hpc_done = ori_done.clone().detach()
    hpc_weight = ori_weight.clone().detach()
    hpc_qntd = QNStepTD(T, B, N)

    if use_cuda:
        ori_q = ori_q.cuda()
        ori_next_n_q = ori_next_n_q.cuda()
        ori_action = ori_action.cuda()
        ori_next_n_action = ori_next_n_action.cuda()
        ori_reward = ori_reward.cuda()
        ori_done = ori_done.cuda()
        ori_weight = ori_weight.cuda()

        hpc_q = hpc_q.cuda()
        hpc_next_n_q = hpc_next_n_q.cuda()
        hpc_action = hpc_action.cuda()
        hpc_next_n_action = hpc_next_n_action.cuda()
        hpc_reward = hpc_reward.cuda()
        hpc_done = hpc_done.cuda()
        hpc_weight = hpc_weight.cuda()
        hpc_qntd = hpc_qntd.cuda()

    ori_q.requires_grad_(True)
    ori_loss, _ = q_nstep_td_error(
        q_nstep_td_data(ori_q, ori_next_n_q, ori_action, ori_next_n_action, ori_reward, ori_done, ori_weight), gamma, T
    )
    ori_loss = ori_loss.mean()
    ori_loss.backward()
    if use_cuda:
        torch.cuda.synchronize()

    hpc_q.requires_grad_(True)
    hpc_loss, _ = hpc_qntd(hpc_q, hpc_next_n_q, hpc_action, hpc_next_n_action, hpc_reward, hpc_done, hpc_weight, gamma)
    hpc_loss = hpc_loss.mean()
    hpc_loss.backward()
    if use_cuda:
        torch.cuda.synchronize()

    mre = mean_relative_error(
        torch.flatten(ori_loss).cpu().detach().numpy(),
        torch.flatten(hpc_loss).cpu().detach().numpy()
    )
    print("qntd fp mean_relative_error: " + str(mre))
    mre = mean_relative_error(
        torch.flatten(ori_q.grad).cpu().detach().numpy(),
        torch.flatten(hpc_q.grad).cpu().detach().numpy()
    )
    print("qntd bp mean_relative_error: " + str(mre))


def qntd_perf():
    ori_q = torch.randn(B, N)
    ori_next_n_q = torch.randn(B, N)
    ori_action = torch.randint(0, N, size=(B, ))
    ori_next_n_action = torch.randint(0, N, size=(B, ))
    ori_reward = torch.randn(T, B)
    ori_done = torch.randn(B)
    ori_weight = torch.randn(B)

    hpc_q = ori_q.clone().detach()
    hpc_next_n_q = ori_next_n_q.clone().detach()
    hpc_action = ori_action.clone().detach()
    hpc_next_n_action = ori_next_n_action.clone().detach()
    hpc_reward = ori_reward.clone().detach()
    hpc_done = ori_done.clone().detach()
    hpc_weight = ori_weight.clone().detach()
    hpc_qntd = QNStepTD(T, B, N)

    if use_cuda:
        ori_q = ori_q.cuda()
        ori_next_n_q = ori_next_n_q.cuda()
        ori_action = ori_action.cuda()
        ori_next_n_action = ori_next_n_action.cuda()
        ori_reward = ori_reward.cuda()
        ori_done = ori_done.cuda()
        ori_weight = ori_weight.cuda()

        hpc_q = hpc_q.cuda()
        hpc_next_n_q = hpc_next_n_q.cuda()
        hpc_action = hpc_action.cuda()
        hpc_next_n_action = hpc_next_n_action.cuda()
        hpc_reward = hpc_reward.cuda()
        hpc_done = hpc_done.cuda()
        hpc_weight = hpc_weight.cuda()
        hpc_qntd = hpc_qntd.cuda()

    ori_q.requires_grad_(True)
    for i in range(times):
        t = time.time()
        ori_loss, _ = q_nstep_td_error(
            q_nstep_td_data(ori_q, ori_next_n_q, ori_action, ori_next_n_action, ori_reward, ori_done, ori_weight),
            gamma, T
        )
        ori_loss = ori_loss.mean()
        ori_loss.backward()
        if use_cuda:
            torch.cuda.synchronize()
        print('epoch: {}, original qntd cost time: {}'.format(i, time.time() - t))

    hpc_q.requires_grad_(True)
    for i in range(times):
        t = time.time()
        hpc_loss, _ = hpc_qntd(
            hpc_q, hpc_next_n_q, hpc_action, hpc_next_n_action, hpc_reward, hpc_done, hpc_weight, gamma
        )
        hpc_loss = hpc_loss.mean()
        hpc_loss.backward()
        if use_cuda:
            torch.cuda.synchronize()
        print('epoch: {}, hpc qntd cost time: {}'.format(i, time.time() - t))


if __name__ == '__main__':
    print("target problem: T = {}, B = {}, N = {}, gamma = {}".format(T, B, N, gamma))
    print("================run qntd validation test================")
    qntd_val()
    print("================run qntd performance test================")
    qntd_perf()
