import time
import torch
import torch.nn.functional as F
from hpc_rll.origin.ppo import ppo_error, ppo_data
from hpc_rll.rl_utils.ppo import PPO
from testbase import mean_relative_error, times

assert torch.cuda.is_available()
use_cuda = True

B = 128
N = 128
clip_ratio = 0.2
use_value_clip = True
dual_clip = None


def ppo_val():
    ori_logits_new = torch.randn(B, N)
    ori_logits_old = torch.randn(B, N)
    ori_action = torch.randint(0, N, size=(B, ))
    ori_value_new = torch.randn(B)
    ori_value_old = torch.randn(B)
    ori_adv = torch.randn(B)
    ori_return = torch.randn(B)
    ori_weight = torch.randn(B)

    hpc_logits_new = ori_logits_new.clone().detach()
    hpc_logits_old = ori_logits_old.clone().detach()
    hpc_action = ori_action.clone().detach()
    hpc_value_new = ori_value_new.clone().detach()
    hpc_value_old = ori_value_old.clone().detach()
    hpc_adv = ori_adv.clone().detach()
    hpc_return = ori_return.clone().detach()
    hpc_weight = ori_weight.clone().detach()
    hpc_ppo = PPO(B, N)

    if use_cuda:
        ori_logits_new = ori_logits_new.cuda()
        ori_logits_old = ori_logits_old.cuda()
        ori_action = ori_action.cuda()
        ori_value_new = ori_value_new.cuda()
        ori_value_old = ori_value_old.cuda()
        ori_adv = ori_adv.cuda()
        ori_return = ori_return.cuda()
        ori_weight = ori_weight.cuda()

        hpc_logits_new = hpc_logits_new.cuda()
        hpc_logits_old = hpc_logits_old.cuda()
        hpc_action = hpc_action.cuda()
        hpc_value_new = hpc_value_new.cuda()
        hpc_value_old = hpc_value_old.cuda()
        hpc_adv = hpc_adv.cuda()
        hpc_return = hpc_return.cuda()
        hpc_weight = hpc_weight.cuda()
        hpc_ppo = hpc_ppo.cuda()

    ori_logits_new.requires_grad_(True)
    ori_value_new.requires_grad_(True)
    ori_loss, ori_info = ppo_error(
        ppo_data(
            ori_logits_new, ori_logits_old, ori_action, ori_value_new, ori_value_old, ori_adv, ori_return, ori_weight
        ), clip_ratio, use_value_clip, dual_clip
    )
    ori_loss = sum(ori_loss)
    ori_loss.backward()

    hpc_logits_new.requires_grad_(True)
    hpc_value_new.requires_grad_(True)
    hpc_loss, hpc_info = hpc_ppo(
        hpc_logits_new, hpc_logits_old, hpc_action, hpc_value_new, hpc_value_old, hpc_adv, hpc_return, hpc_weight,
        clip_ratio, use_value_clip, dual_clip
    )
    hpc_loss = sum(hpc_loss)
    hpc_loss.backward()

    print("ori_info: " + str(ori_info))
    print("hpc_info: " + str(hpc_info))
    mre = mean_relative_error(
        torch.flatten(ori_loss).cpu().detach().numpy(),
        torch.flatten(hpc_loss).cpu().detach().numpy()
    )
    print("ppo fp loss mean_relative_error: " + str(mre))
    mre = mean_relative_error(
        torch.flatten(ori_logits_new.grad).cpu().detach().numpy(),
        torch.flatten(hpc_logits_new.grad).cpu().detach().numpy()
    )
    print("ppo bp logits_new mean_relative_error: " + str(mre))
    mre = mean_relative_error(
        torch.flatten(ori_value_new.grad).cpu().detach().numpy(),
        torch.flatten(hpc_value_new.grad).cpu().detach().numpy()
    )
    print("ppo bp value_new mean_relative_error: " + str(mre))


def ppo_perf():
    ori_logits_new = torch.randn(B, N)
    ori_logits_old = torch.randn(B, N)
    ori_action = torch.randint(0, N, size=(B, ))
    ori_value_new = torch.randn(B)
    ori_value_old = torch.randn(B)
    ori_adv = torch.randn(B)
    ori_return = torch.randn(B)
    ori_weight = torch.randn(B)

    hpc_logits_new = ori_logits_new.clone().detach()
    hpc_logits_old = ori_logits_old.clone().detach()
    hpc_action = ori_action.clone().detach()
    hpc_value_new = ori_value_new.clone().detach()
    hpc_value_old = ori_value_old.clone().detach()
    hpc_adv = ori_adv.clone().detach()
    hpc_return = ori_return.clone().detach()
    hpc_weight = ori_weight.clone().detach()
    hpc_ppo = PPO(B, N)

    if use_cuda:
        ori_logits_new = ori_logits_new.cuda()
        ori_logits_old = ori_logits_old.cuda()
        ori_action = ori_action.cuda()
        ori_value_new = ori_value_new.cuda()
        ori_value_old = ori_value_old.cuda()
        ori_adv = ori_adv.cuda()
        ori_return = ori_return.cuda()
        ori_weight = ori_weight.cuda()

        hpc_logits_new = hpc_logits_new.cuda()
        hpc_logits_old = hpc_logits_old.cuda()
        hpc_action = hpc_action.cuda()
        hpc_value_new = hpc_value_new.cuda()
        hpc_value_old = hpc_value_old.cuda()
        hpc_adv = hpc_adv.cuda()
        hpc_return = hpc_return.cuda()
        hpc_weight = hpc_weight.cuda()
        hpc_ppo = hpc_ppo.cuda()

    ori_logits_new.requires_grad_(True)
    ori_value_new.requires_grad_(True)
    for i in range(times):
        t = time.time()
        ori_loss, ori_info = ppo_error(
            ppo_data(
                ori_logits_new, ori_logits_old, ori_action, ori_value_new, ori_value_old, ori_adv, ori_return,
                ori_weight
            ), clip_ratio, use_value_clip, dual_clip
        )
        ori_loss = sum(ori_loss)
        ori_loss.backward()
        if use_cuda:
            torch.cuda.synchronize()
        print('epoch: {}, origin ppo cost time: {}'.format(i, time.time() - t))

    hpc_logits_new.requires_grad_(True)
    hpc_value_new.requires_grad_(True)
    for i in range(times):
        t = time.time()
        hpc_loss, hpc_info = hpc_ppo(
            hpc_logits_new, hpc_logits_old, hpc_action, hpc_value_new, hpc_value_old, hpc_adv, hpc_return, hpc_weight,
            clip_ratio, use_value_clip, dual_clip
        )
        hpc_loss = sum(hpc_loss)
        hpc_loss.backward()
        if use_cuda:
            torch.cuda.synchronize()
        print('epoch: {}, hpc ppo cost time: {}'.format(i, time.time() - t))


if __name__ == '__main__':
    print(
        "target problem: B = {}, N = {}, clip_ratio = {}, use_value_clip = {}, dual_clip = {}".format(
            B, N, clip_ratio, use_value_clip, dual_clip
        )
    )
    print("================run ppo validation test================")
    ppo_val()
    print("================run ppo performance test================")
    ppo_perf()
