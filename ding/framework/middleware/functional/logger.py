import os
import wandb
import torch
import numpy as np
from typing import TYPE_CHECKING, Callable, Union
from ding.envs import BaseEnvManagerV2
from easydict import EasyDict
from ding.framework import task
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import plotly.express as px

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