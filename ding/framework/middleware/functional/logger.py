from typing import TYPE_CHECKING, Callable, Dict, List, Union
import numpy as np
import os
import wandb
from matplotlib import pyplot as plt
from matplotlib import animation
from ding.envs import BaseEnvManagerV2
from easydict import EasyDict
from ding.utils import DistributedWriter
from ding.torch_utils import to_ndarray
from torch.nn import functional as F
from ding.utils.default_helper import one_time_warning

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


def wandb_logger(cfg: EasyDict, env: BaseEnvManagerV2, model) -> Callable:
    color_list = ["orange", "red", "blue", "purple", "green", "darkcyan"]
    metric_list = ["q_value", "target q_value", "loss", "lr", "entropy"]
    env.enable_save_replay(replay_path=cfg.record_path)
    if cfg.gradient_logger:
        wandb.watch(model)
    else:
        one_time_warning(
            "If you want to use wandb to visualize the gradient, please set gradient_logger = True in the config."
        )

    def _action_prob(num, action_prob, ln):
        ax = plt.gca()
        ax.set_ylim([0, 1])
        for rect, x in zip(ln, action_prob[num]):
            rect.set_height(x)
        return ln

    def _value_prob(num, action_prob, action, ln):
        ax = plt.gca()
        ax.set_ylim([0, 1])
        for rect, x in zip(ln, action_prob[num][action[num][0] - 1]):
            rect.set_height(x)
        return ln

    def _plot(ctx: "OnlineRLContext"):
        if not cfg.plot_logger:
            one_time_warning(
                "If you want to use wandb to visualize the result, please set plot_logger = True in the config."
            )
            return
        for metric in metric_list:
            if metric in ctx.train_output[0]:
                metric_value = np.mean([item[metric] for item in ctx.train_output])
                wandb.log({metric: metric_value})
        if ctx.eval_value != -np.inf:
            wandb.log({"reward": ctx.eval_value})

        if ctx.eval_value != -np.inf:
            if 'logit' in ctx.eval_output[0]:
                action_value = [to_ndarray(F.softmax(v['logit'], dim=-1)) for v in ctx.eval_output]
            if 'distribution' in ctx.eval_output[0]:
                value_dist = [to_ndarray(v['distribution']) for v in ctx.eval_output]
                action = [to_ndarray(v['action']) for v in ctx.eval_output]

            file_list = []
            for p in os.listdir(cfg.record_path):
                if os.path.splitext(p)[-1] == ".mp4":
                    file_list.append(p)
            file_list.sort(key=lambda fn: os.path.getmtime(os.path.join(cfg.record_path, fn)))

            fig, ax = plt.subplots()
            plt.ylim([-1, 1])
            if cfg.action_logger == 'q_value' or 'action probability':
                action_dim = len(action_value[0])
                x_range = [str(x + 1) for x in range(action_dim)]
                ln = ax.bar(x_range, [0 for x in range(action_dim)], color=color_list[:action_dim])
                ani = animation.FuncAnimation(
                    fig, _action_prob, fargs=(action_value, ln), blit=True, save_count=len(action_value)
                )
                ani.save(os.path.join(cfg.record_path, (str(ctx.env_step) + ".gif")), writer='pillow')
                wandb.log(
                    {
                        "video": wandb.Video(os.path.join(cfg.record_path, file_list[-2]), format="mp4"),
                        "q value": wandb.Video(
                            os.path.join(cfg.record_path, (str(ctx.env_step) + ".gif")), format="gif"
                        )
                    }
                )
            elif cfg.action_logger == 'q_value distribution':
                action_dim = len(value_dist[0])
                dist_dim = len(value_dist[0][0])
                assert action_dim == model.action_shape
                x_range = [str(x + 1) for x in range(dist_dim)]
                ln = ax.bar(x_range, [0 for x in range(dist_dim)], color='r')
                ani = animation.FuncAnimation(
                    fig, _value_prob, fargs=(value_dist, action, ln), blit=True, save_count=len(value_dist)
                )
                ani.save(os.path.join(cfg.record_path, (str(ctx.env_step) + ".gif")), writer='pillow')
                wandb.log(
                    {
                        "video": wandb.Video(os.path.join(cfg.record_path, file_list[-2]), format="mp4"),
                        "q value distribution": wandb.Video(
                            os.path.join(cfg.record_path, (str(ctx.env_step) + ".gif")), format="gif"
                        )
                    }
                )

    return _plot
