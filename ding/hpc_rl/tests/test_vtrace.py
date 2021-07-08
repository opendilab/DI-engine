import time
import torch
import torch.nn.functional as F
from hpc_rll.origin.vtrace import vtrace_error, vtrace_data
from hpc_rll.rl_utils.vtrace import VTrace
from testbase import mean_relative_error, times

assert torch.cuda.is_available()
use_cuda = True

T = 128
B = 128
N = 128


def vtrace_val():
    ori_target_output = torch.randn(T, B, N)
    ori_behaviour_output = torch.randn(T, B, N)
    ori_action = torch.randint(
        0, N, size=(
            T,
            B,
        )
    )
    ori_value = torch.randn(T + 1, B)
    ori_reward = torch.randn(T, B)

    hpc_target_output = ori_target_output.clone().detach()
    hpc_behaviour_output = ori_behaviour_output.clone().detach()
    hpc_action = ori_action.clone().detach()
    hpc_value = ori_value.clone().detach()
    hpc_reward = ori_reward.clone().detach()
    hpc_vtrace = VTrace(T, B, N)

    if use_cuda:
        ori_target_output = ori_target_output.cuda()
        ori_behaviour_output = ori_behaviour_output.cuda()
        ori_action = ori_action.cuda()
        ori_value = ori_value.cuda()
        ori_reward = ori_reward.cuda()

        hpc_target_output = hpc_target_output.cuda()
        hpc_behaviour_output = hpc_behaviour_output.cuda()
        hpc_action = hpc_action.cuda()
        hpc_value = hpc_value.cuda()
        hpc_reward = hpc_reward.cuda()
        hpc_vtrace = hpc_vtrace.cuda()

    ori_target_output.requires_grad_(True)
    ori_value.requires_grad_(True)
    ori_loss = vtrace_error(
        vtrace_data(ori_target_output, ori_behaviour_output, ori_action, ori_value, ori_reward, None)
    )
    ori_loss = sum(ori_loss)
    ori_loss.backward()

    hpc_target_output.requires_grad_(True)
    hpc_value.requires_grad_(True)
    hpc_loss = hpc_vtrace(hpc_target_output, hpc_behaviour_output, hpc_action, hpc_value, hpc_reward)
    hpc_loss = sum(hpc_loss)
    hpc_loss.backward()

    mre = mean_relative_error(
        torch.flatten(ori_loss).cpu().detach().numpy(),
        torch.flatten(hpc_loss).cpu().detach().numpy()
    )
    print("vtrace fp mean_relative_error: " + str(mre))
    mre = mean_relative_error(
        torch.flatten(ori_target_output.grad).cpu().detach().numpy(),
        torch.flatten(hpc_target_output.grad).cpu().detach().numpy()
    )
    print("vtrace bp target_output mean_relative_error: " + str(mre))
    mre = mean_relative_error(
        torch.flatten(ori_value.grad).cpu().detach().numpy(),
        torch.flatten(hpc_value.grad).cpu().detach().numpy()
    )
    print("vtrace bp value mean_relative_error: " + str(mre))


def vtrace_perf():
    ori_target_output = torch.randn(T, B, N)
    ori_behaviour_output = torch.randn(T, B, N)
    ori_action = torch.randint(
        0, N, size=(
            T,
            B,
        )
    )
    ori_value = torch.randn(T + 1, B)
    ori_reward = torch.randn(T, B)

    hpc_target_output = ori_target_output.clone().detach()
    hpc_behaviour_output = ori_behaviour_output.clone().detach()
    hpc_action = ori_action.clone().detach()
    hpc_value = ori_value.clone().detach()
    hpc_reward = ori_reward.clone().detach()
    hpc_vtrace = VTrace(T, B, N)

    if use_cuda:
        ori_target_output = ori_target_output.cuda()
        ori_behaviour_output = ori_behaviour_output.cuda()
        ori_action = ori_action.cuda()
        ori_value = ori_value.cuda()
        ori_reward = ori_reward.cuda()

        hpc_target_output = hpc_target_output.cuda()
        hpc_behaviour_output = hpc_behaviour_output.cuda()
        hpc_action = hpc_action.cuda()
        hpc_value = hpc_value.cuda()
        hpc_reward = hpc_reward.cuda()
        hpc_vtrace = hpc_vtrace.cuda()

    ori_target_output.requires_grad_(True)
    ori_value.requires_grad_(True)
    for i in range(times):
        t = time.time()
        ori_loss = vtrace_error(
            vtrace_data(ori_target_output, ori_behaviour_output, ori_action, ori_value, ori_reward, None)
        )
        ori_loss = sum(ori_loss)
        ori_loss.backward()
        if use_cuda:
            torch.cuda.synchronize()
        print('epoch: {}, original vtrace cost time: {}'.format(i, time.time() - t))

    hpc_target_output.requires_grad_(True)
    hpc_value.requires_grad_(True)
    for i in range(times):
        t = time.time()
        hpc_loss = hpc_vtrace(hpc_target_output, hpc_behaviour_output, hpc_action, hpc_value, hpc_reward)
        hpc_loss = sum(hpc_loss)
        hpc_loss.backward()
        if use_cuda:
            torch.cuda.synchronize()
        print('epoch: {}, hpc vtrace cost time: {}'.format(i, time.time() - t))


if __name__ == '__main__':
    print("target problem: T = {}, B = {}, N = {}".format(T, B, N))
    print("================run vtrace validation test================")
    vtrace_val()
    print("================run vtrace performance test================")
    vtrace_perf()
