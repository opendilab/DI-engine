import os
import copy
import torch
from torch.distributions import Normal
import logging
from functools import partial
from tensorboardX import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt

from ding.utils import set_pkg_seed, read_file, save_file

envstep_1 = 25500
envstep_2 = 48750
exp_name = 'sac_hopper_mopo_default_config' + '_{}_{}'.format(envstep_1, envstep_2)
profile_path = exp_name + '.pth.tar'

dict = read_file(profile_path)
obs = dict['obs'].cpu()
std =  obs.std(0)
next_obs_pred = dict['next_obs_pred'].cpu()
next_obs_real = dict['next_obs_real'].cpu()

reward_pred = dict['reward_pred'].cpu()
reward_real = dict['reward_real'].cpu()

next_q_pred = dict['next_q_pred'].cpu()
next_q_real = dict['next_q_real'].cpu()

mu = dict['dist'][0].cpu()
sigma = dict['dist'][1].cpu()
q = dict['q'].cpu()
mu_new = dict['dist_new'][0].cpu()
sigma_new = dict['dist_new'][1].cpu()
q_new = dict['q_new'].cpu()

batch_size, state_dim, action_dim, n_actions = obs.shape[0], obs.shape[1], mu.shape[1], q.shape[-1]

def sample(mu, sigma):
    dist = Normal(mu, sigma)
    x = dist.sample((1000,))
    x, _ = x.sort()
    y = torch.tanh(x)
    log_prob = dist.log_prob(x) + 2 * torch.log(torch.cosh(x))
    prob = torch.exp(log_prob)
    return y, prob

def plot(sample_id, action_id):
    # dist
    mu_, sigma_ = mu[sample_id][action_id], sigma[sample_id][action_id]
    mu_new_, sigma_new_ = mu_new[sample_id][action_id], sigma_new[sample_id][action_id]
    action, prob = sample(mu_, sigma_)
    action_new, prob_new = sample(mu_new_, sigma_new_)

    # q
    q_value = q[sample_id][action_id]
    q_value_new = q_new[sample_id][action_id]
    q_value = q_value - q_value.mean()
    q_value_new = q_value_new - q_value_new.mean()

    # model
    next_q_pred_ = next_q_pred[sample_id][action_id]
    next_q_real_ = next_q_real[sample_id][action_id]
    reward_pred_ = reward_pred[sample_id][action_id]
    reward_real_ = reward_real[sample_id][action_id]
    next_obs_pred_ = next_obs_pred[sample_id][action_id]
    next_obs_real_ = next_obs_real[sample_id][action_id]
    error = torch.abs(next_obs_pred_ - next_obs_real_) / std.unsqueeze(0)
    error = error.mean(1)
    act = torch.arange(n_actions).float() / (n_actions - 1) * 2 - 1

    # draw
    fig = plt.figure()

    ax = fig.add_subplot(2,1,1)
    ax2 = ax.twinx()
    ax.plot(action.numpy(), prob.numpy(), 'r-', label = 'dist_old')
    ax.plot(action_new.numpy(), prob_new.numpy(), 'b-', label = 'dist_new')
    ax2.plot(act.numpy(), q_value.numpy(), 'r:', label = 'adv_old')
    ax2.plot(act.numpy(), q_value_new.numpy(), 'b:', label = 'adv_new')
    ax.set_xlabel("action")
    ax.set_ylabel("dist")
    ax2.set_ylabel("q value")

    ax = fig.add_subplot(2,1,2)
    ax2 = ax.twinx()
    ax.plot(act.numpy(), error.numpy(), 'r-', label = 'error')
    ax2.plot(act.numpy(), reward_pred_.numpy() + 0.99 * next_q_pred_.numpy(), 'r:', label = 'target_q_pred')
    ax2.plot(act.numpy(), reward_real_.numpy() + 0.99 * next_q_real_.numpy(), 'b:', label = 'target_q_real')
    ax.set_xlabel("action")
    ax.set_ylabel("error")
    # ax.set_ylim(error.min(), error.max())
    ax2.set_ylabel("next q value")
    plt.savefig('sample_{}_action_{}.png'.format(sample_id, action_id))
    plt.close()

for sample_id in range(batch_size):
    for action_id in range(3):
        plot(sample_id, action_id)
        print("plot {}-th sample {}-th action dim".format(sample_id, action_id))
