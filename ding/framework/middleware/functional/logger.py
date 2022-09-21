from typing import TYPE_CHECKING, Callable, Dict, List， Union
import numpy as np
import os
import wandb
import torch
from ding.envs import BaseEnvManagerV2
from easydict import EasyDict
from ding.framework import task
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import plotly.express as px
from ding.utils import DistributedWriter

if TYPE_CHECKING:
    from ding.framework import OnlineRLContext, OfflineRLContext
    
def online_logger(record_train_iter: bool = False, train_show_freq: int = 100) -> Callable:
    writer = DistributedWriter.get_instance()
    last_train_show_iter = -1

    def _logger(ctx: "OnlineRLContext"):
        nonlocal last_train_show_iter
        if not np.isinf(ctx.eval_value):
            if record_train_iter:
                writer.add_scalar('basic/eval_episode_reward_mean-env_step', ctx.eval_value, ctx.env_step)
                writer.add_scalar('basic/eval_episode_reward_mean-train_iter', ctx.eval_value, ctx.train_iter)
            else:
                writer.add_scalar('basic/eval_episode_reward_mean', ctx.eval_value, ctx.env_step)
        if ctx.train_output is not None and ctx.train_iter - last_train_show_iter >= train_show_freq:
            last_train_show_iter = ctx.train_iter
            if isinstance(ctx.train_output, List):
                output = ctx.train_output.pop()  # only use latest output
            else:
                output = ctx.train_output
            for k, v in output.items():
                if k in ['priority']:
                    continue
                if "[scalars]" in k:
                    new_k = k.split(']')[-1]
                    raise NotImplementedError
                elif "[histogram]" in k:
                    new_k = k.split(']')[-1]
                    writer.add_histogram(new_k, v, ctx.env_step)
                    if record_train_iter:
                        writer.add_histogram(new_k, v, ctx.train_iter)
                else:
                    if record_train_iter:
                        writer.add_scalar('basic/train_{}-train_iter'.format(k), v, ctx.train_iter)
                        writer.add_scalar('basic/train_{}-env_step'.format(k), v, ctx.env_step)
                    else:
                        writer.add_scalar('basic/train_{}'.format(k), v, ctx.env_step)

    return _logger


def offline_logger() -> Callable:
    writer = DistributedWriter.get_instance()

    def _logger(ctx: "OfflineRLContext"):
        if not np.isinf(ctx.eval_value):
            writer.add_scalar('basic/eval_episode_reward_mean-train_iter', ctx.eval_value, ctx.train_iter)
        if ctx.train_output is not None:
            output = ctx.train_output
            for k, v in output.items():
                if k in ['priority']:
                    continue
                if "[scalars]" in k:
                    new_k = k.split(']')[-1]
                    raise NotImplementedError
                elif "[histogram]" in k:
                    new_k = k.split(']')[-1]
                    writer.add_histogram(new_k, v, ctx.train_iter)
                else:
                    writer.add_scalar('basic/train_{}-train_iter'.format(k), v, ctx.train_iter)

    return _logger

def wandb_logger(cfg:EasyDict, env:BaseEnvManagerV2, model) -> Callable:
    
    color_list = ["orange", "red", "blue", "purple", "green", "darkcyan"]
    if cfg.policy.logger.gradient_logger:
        wandb.watch(model)

    def _action_prob(num, action_prob, ln):
        ax = plt.gca()
        ax.set_ylim([0, 1])
        for rect, x in zip(ln, action_prob[num][0]):
            rect.set_height(x)
        return ln

    def _value_prob(num, action_prob, action, ln):
        ax = plt.gca()
        ax.set_ylim([0, 1])
        for rect, x in zip(ln, action_prob[num][action[num][0]-1]):
            rect.set_height(x)
        return ln

    def _plot(ctx: "OnlineRLContext"):

        if cfg.policy.logger.plot_logger.q_value:
            q_value = torch.mean(torch.tensor([item['q_value'] for item in ctx.train_output]))
            wandb.log({"q_value":q_value})
        if cfg.policy.logger.plot_logger.target_q_value:
            target_q_value = torch.mean(torch.tensor([item['target_q_value'] for item in ctx.train_output]))
            wandb.log({"target q_value":target_q_value})
        if cfg.policy.logger.plot_logger.loss:
            loss = torch.mean(torch.tensor([item['total_loss'] for item in ctx.train_output]))
            wandb.log({"loss":loss})
        if cfg.policy.logger.plot_logger.lr:
            lr = torch.mean(torch.tensor([item['cur_lr'] for item in ctx.train_output]))
            wandb.log({"lr":lr})
        if cfg.policy.logger.plot_logger.entropy:
            entropy = torch.mean(torch.tensor([item['entropy'] for item in ctx.train_output]))
            wandb.log({"entropy":entropy})
        if ctx.eval_value != -np.inf:
            wandb.log({"reward":ctx.eval_value})

    
        if ctx.eval_value != -np.inf:
            file_list = []
            for p in os.listdir(cfg.policy.log.record_path):
                if os.path.splitext(p)[-1] == ".mp4":
                    file_list.append(p)
            file_list.sort(key=lambda fn:os.path.getmtime(cfg.policy.log.record_path+"/"+fn))

            fig, ax = plt.subplots()
            plt.ylim([-1, 1])
            if cfg.policy.logger.action_logger == 'q_value' or 'action probability':
                action_dim = len(ctx.eval_action_prob[0][0])
                x_range = [str(x+1) for x in range(action_dim)]
                ln= ax.bar(x_range, [0 for x in range(action_dim)], color = color_list[:action_dim])
                ani = animation.FuncAnimation(fig, _action_prob, fargs=(ctx.eval_action_prob, ln), blit=True, save_count=len(ctx.eval_action_prob))
                ani.save(cfg.policy.log.record_path+"/"+str(ctx.env_step) + ".gif", writer='pillow')
                wandb.log({"video": wandb.Video(cfg.policy.log.record_path+"/"+file_list[-2], format="mp4"), "q value": wandb.Video(cfg.policy.log.record_path+"/"+str(ctx.env_step) + ".gif", format="gif")})
            if cfg.policy.logger.action_logger == 'q_value distribution':
                action_dim = len(ctx.eval_value_dist[0])
                dist_dim = len(ctx.eval_value_dist[0][0])
                x_range = [str(x+1) for x in range(dist_dim)]
                ln= ax.bar(x_range, [0 for x in range(dist_dim)], color = 'r')
                ani = animation.FuncAnimation(fig, _value_prob, fargs=(ctx.eval_value_dist, ctx.eval_action, ln), blit=True, save_count=len(ctx.eval_value_dist))
                ani.save(cfg.policy.log.record_path+"/"+str(ctx.env_step) + ".gif", writer='pillow')
                wandb.log({"video": wandb.Video(cfg.policy.log.record_path+"/"+file_list[-2], format="mp4"), "q value distribution": wandb.Video(cfg.policy.log.record_path+"/"+str(ctx.env_step) + ".gif", format="gif")})
    return _plot
