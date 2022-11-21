from typing import TYPE_CHECKING, Callable, Dict, List, Union
import os
from easydict import EasyDict
from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib import ticker as mtick
from torch.nn import functional as F
from sklearn.manifold import TSNE
import numpy as np
import torch
import wandb
import h5py
import pickle
from ding.envs import BaseEnvManagerV2
from ding.utils import DistributedWriter
from ding.torch_utils import to_ndarray
from ding.utils.default_helper import one_time_warning

if TYPE_CHECKING:
    from ding.framework import OnlineRLContext, OfflineRLContext


def action_prob(num, action_prob, ln):
    ax = plt.gca()
    ax.set_ylim([0, 1])
    for rect, x in zip(ln, action_prob[num]):
        rect.set_height(x)
    return ln


def return_prob(num, return_prob, ln):
    return ln


def return_distribution(reward):
    num = len(reward)
    max_return = max(reward)
    min_return = min(reward)
    hist, bins = np.histogram(reward, bins=np.linspace(min_return - 50, max_return + 50, 6))
    gap = (max_return - min_return + 100) / 5
    x_dim = ['{:.1f}'.format(min_return - 50 + gap * x) for x in range(5)]
    return hist / num, x_dim


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


def wandb_online_logger(
        cfg: EasyDict, env: BaseEnvManagerV2, model: torch.nn.Module, anonymous: bool = False
) -> Callable:
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
        - anonymous (:obj:`bool`): Open the anonymous mode of wandb or not.
            The anonymous mode allows visualization of data without wandb count.
    '''

    color_list = ["orange", "red", "blue", "purple", "green", "darkcyan"]
    metric_list = ["q_value", "target q_value", "loss", "lr", "entropy"]
    # Initialize wandb with default settings
    # Settings can be covered by calling wandb.init() at the top of the script
    if anonymous:
        wandb.init(anonymous="must")
    else:
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
            wandb.log({"reward": ctx.eval_value, "train iter": ctx.train_iter})

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
            if cfg.action_logger in ['q_value', 'action probability']:
                fig, ax = plt.subplots()
                plt.ylim([-1, 1])
                action_dim = len(action_value[0])
                x_range = [str(x + 1) for x in range(action_dim)]
                ln = ax.bar(x_range, [0 for x in range(action_dim)], color=color_list[:action_dim])
                ani = animation.FuncAnimation(
                    fig, action_prob, fargs=(action_value, ln), blit=True, save_count=len(action_value)
                )
                ani.save(action_path, writer='pillow')
                wandb.log({cfg.action_logger: wandb.Video(action_path, format="gif")})
                plt.clf()

            fig, ax = plt.subplots()
            ax = plt.gca()
            ax.set_ylim([0, 1])
            hist, x_dim = return_distribution(eval_reward)
            assert len(hist) == len(x_dim)
            ln_return = ax.bar(x_dim, hist, width=1, color='r', linewidth=0.7)
            ani = animation.FuncAnimation(fig, return_prob, fargs=(hist, ln_return), blit=True, save_count=1)
            ani.save(return_path, writer='pillow')
            wandb.log(
                {
                    "video": wandb.Video(video_path, format="mp4"),
                    "return distribution": wandb.Video(return_path, format="gif")
                }
            )

    return _plot


def wandb_offline_logger(
        cfg: EasyDict,
        env: BaseEnvManagerV2,
        model: torch.nn.Module,
        datasetpath: str,
        anonymous: bool = False
) -> Callable:
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
        - datasetpath (:obj:`str`): The path of offline dataset.
        - anonymous (:obj:`bool`): Open the anonymous mode of wandb or not.
            The anonymous mode allows visualization of data without wandb count.
    '''

    color_list = ["orange", "red", "blue", "purple", "green", "darkcyan"]
    metric_list = ["q_value", "target q_value", "loss", "lr", "entropy", "target_q_value", "td_error"]
    # Initialize wandb with default settings
    # Settings can be covered by calling wandb.init() at the top of the script
    if anonymous:
        wandb.init(anonymous="must")
    else:
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

    def _vis_dataset(datasetpath: str):
        assert os.path.splitext(datasetpath)[-1] in ['.pkl', '.h5', '.hdf5']
        if os.path.splitext(datasetpath)[-1] == '.pkl':
            with open(datasetpath, 'rb') as f:
                data = pickle.load(f)
            obs = []
            action = []
            reward = []
            for i in range(len(data)):
                obs.extend(data[i]['observations'])
                action.extend(data[i]['actions'])
                reward.extend(data[i]['rewards'])
        elif os.path.splitext(datasetpath)[-1] in ['.h5', '.hdf5']:
            with h5py.File(datasetpath, 'r') as f:
                obs = f['obs'][()]
                action = f['action'][()]
                reward = f['reward'][()]

        cmap = plt.cm.hsv
        obs = np.array(obs)
        reward = np.array(reward)
        obs_action = np.hstack((obs, np.array(action)))
        reward = reward / (max(reward) - min(reward))

        embedded_obs = TSNE(n_components=2).fit_transform(obs)
        embedded_obs_action = TSNE(n_components=2).fit_transform(obs_action)
        x_min, x_max = np.min(embedded_obs, 0), np.max(embedded_obs, 0)
        embedded_obs = embedded_obs / (x_max - x_min)

        x_min, x_max = np.min(embedded_obs_action, 0), np.max(embedded_obs_action, 0)
        embedded_obs_action = embedded_obs_action / (x_max - x_min)

        fig = plt.figure()
        f, axes = plt.subplots(nrows=1, ncols=3)

        axes[0].scatter(embedded_obs[:, 0], embedded_obs[:, 1], c=cmap(reward))
        axes[1].scatter(embedded_obs[:, 0], embedded_obs[:, 1], c=cmap(action))
        axes[2].scatter(embedded_obs_action[:, 0], embedded_obs_action[:, 1], c=cmap(reward))
        axes[0].set_title('state-reward')
        axes[1].set_title('state-action')
        axes[2].set_title('stateAction-reward')
        plt.savefig('dataset.png')

        wandb.log({"dataset": wandb.Image("dataset.png")})

    if cfg.vis_dataset is True:
        _vis_dataset(datasetpath)

    def _plot(ctx: "OfflineRLContext"):
        if not cfg.plot_logger:
            one_time_warning(
                "If you want to use wandb to visualize the result, please set plot_logger = True in the config."
            )
            return
        for metric in metric_list:
            if metric in ctx.train_output:
                metric_value = ctx.train_output[metric]
                wandb.log({metric: metric_value})

        if ctx.eval_value != -np.inf:
            wandb.log({"reward": ctx.eval_value, "train iter": ctx.train_iter})

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
            action_path = os.path.join(cfg.record_path, (str(ctx.train_iter) + "_action.gif"))
            return_path = os.path.join(cfg.record_path, (str(ctx.train_iter) + "_return.gif"))
            if cfg.action_logger in ['q_value', 'action probability']:
                fig, ax = plt.subplots()
                plt.ylim([-1, 1])
                action_dim = len(action_value[0])
                x_range = [str(x + 1) for x in range(action_dim)]
                ln = ax.bar(x_range, [0 for x in range(action_dim)], color=color_list[:action_dim])
                ani = animation.FuncAnimation(
                    fig, action_prob, fargs=(action_value, ln), blit=True, save_count=len(action_value)
                )
                ani.save(action_path, writer='pillow')
                wandb.log({cfg.action_logger: wandb.Video(action_path, format="gif")})
                plt.clf()

            fig, ax = plt.subplots()
            ax = plt.gca()
            ax.set_ylim([0, 1])
            hist, x_dim = return_distribution(eval_reward)
            assert len(hist) == len(x_dim)
            ln_return = ax.bar(x_dim, hist, width=1, color='r', linewidth=0.7)
            ani = animation.FuncAnimation(fig, return_prob, fargs=(hist, ln_return), blit=True, save_count=1)
            ani.save(return_path, writer='pillow')
            wandb.log(
                {
                    "video": wandb.Video(video_path, format="mp4"),
                    "return distribution": wandb.Video(return_path, format="gif")
                }
            )

    return _plot
