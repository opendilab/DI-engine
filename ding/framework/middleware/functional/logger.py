from typing import TYPE_CHECKING, Callable, Dict, List, Union
import os
from easydict import EasyDict
from matplotlib import pyplot as plt
from matplotlib import animation
from torch.nn import functional as F
import numpy as np
import wandb
from ding.envs import BaseEnvManagerV2
from ding.utils import DistributedWriter
from ding.torch_utils import to_ndarray
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


def wandb_online_logger(cfg: EasyDict, env: BaseEnvManagerV2, model) -> Callable:
    '''
    Overview:
        Wandb visualizer to track the experiment.
    Arguments:
        - cfg (:obj:`EasyDict`): Config, a dict of following settings:
            - record_path: string. The path to save the replay of simulation.
            - gradient_logger: boolean. Whether to track the gradient.
            - plot_logger: boolean. Whether to track the metrics like reward and loss.
            - action_logger: `q_value` or `action probability`.
        - env (:obj:`BaseEnvManagerV2`): Evaluator environment.
        - model (:obj:`nn.Module`): Model.
    '''

    color_list = ["orange", "red", "blue", "purple", "green", "darkcyan"]
    metric_list = ["q_value", "target q_value", "loss", "lr", "entropy"]
    # Initialize wandb with default settings
    # Settings can be covered by calling wandb.init() at the top of the script
    wandb.init()
    # The visualizer is called to save the replay of the simulation
    # which will be uploaded to wandb later
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

    def _return_prob(num, return_prob, ln):
        return ln

    def _return_distribution(reward):
        num = len(reward)
        max_return = max(reward)
        min_return = min(reward)
        hist, bins = np.histogram(reward, bins=np.linspace(min_return - 50, max_return + 50, 6))
        gap = (max_return - min_return + 100) / 5
        x_dim = [str(min_return - 50 + gap * x) for x in range(5)]
        return hist / num, x_dim

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

            eval_output = ctx.eval_output['output']
            eval_reward = ctx.eval_output['reward']
            if 'logit' in eval_output[0]:
                action_value = [to_ndarray(F.softmax(v['logit'], dim=-1)) for v in eval_output]

            file_list = []
            for p in os.listdir(cfg.record_path):
                if os.path.splitext(p)[-1] == ".mp4":
                    file_list.append(p)
            file_list.sort(key=lambda fn: os.path.getmtime(os.path.join(cfg.record_path, fn)))

            video_path = os.path.join(cfg.record_path, file_list[-2])
            action_path = os.path.join(cfg.record_path, (str(ctx.env_step) + "_action.gif"))
            return_path = os.path.join(cfg.record_path, (str(ctx.env_step) + "_return.gif"))
            if cfg.action_logger == 'q_value' or 'action probability':
                fig, ax = plt.subplots()
                plt.ylim([-1, 1])
                action_dim = len(action_value[0])
                x_range = [str(x + 1) for x in range(action_dim)]
                ln = ax.bar(x_range, [0 for x in range(action_dim)], color=color_list[:action_dim])
                ani = animation.FuncAnimation(
                    fig, _action_prob, fargs=(action_value, ln), blit=True, save_count=len(action_value)
                )
                ani.save(action_path, writer='pillow')
                wandb.log({cfg.action_logger: wandb.Video(action_path, format="gif")})
                plt.close()

            fig, ax = plt.subplots()
            ax = plt.gca()
            ax.set_ylim([0, 1])
            hist, x_dim = _return_distribution(eval_reward)
            assert len(hist) == len(x_dim)
            ln_return = ax.bar(x_dim, hist, width=1, color='r', linewidth=0.7)
            ani = animation.FuncAnimation(fig, _return_prob, fargs=(hist, ln_return), blit=True, save_count=1)
            ani.save(return_path, writer='pillow')
            wandb.log(
                {
                    "video": wandb.Video(video_path, format="mp4"),
                    "return distribution": wandb.Video(return_path, format="gif")
                }
            )

    return _plot
