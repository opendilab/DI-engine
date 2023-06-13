from typing import TYPE_CHECKING, Optional, Callable, Dict, List, Union
from ditk import logging
from easydict import EasyDict
from matplotlib import pyplot as plt
from matplotlib import animation
import os
import numpy as np
import torch
import wandb
import pickle
import treetensor.numpy as tnp
from ding.framework import task
from ding.envs import BaseEnvManagerV2
from ding.utils import DistributedWriter
from ding.torch_utils import to_ndarray
from ding.utils.default_helper import one_time_warning

if TYPE_CHECKING:
    from ding.framework import OnlineRLContext, OfflineRLContext


def softmax(logit):
    v = np.exp(logit)
    return v / v.sum(axis=-1, keepdims=True)


def action_prob(num, action_prob, ln):
    ax = plt.gca()
    ax.set_ylim([0, 1])
    for rect, x in zip(ln, action_prob[num]):
        rect.set_height(x)
    return ln


def return_prob(num, return_prob, ln):
    return ln


def return_distribution(episode_return):
    num = len(episode_return)
    max_return = max(episode_return)
    min_return = min(episode_return)
    hist, bins = np.histogram(episode_return, bins=np.linspace(min_return - 50, max_return + 50, 6))
    gap = (max_return - min_return + 100) / 5
    x_dim = ['{:.1f}'.format(min_return - 50 + gap * x) for x in range(5)]
    return hist / num, x_dim


def online_logger(record_train_iter: bool = False, train_show_freq: int = 100) -> Callable:
    if task.router.is_active and not task.has_role(task.role.LEARNER):
        return task.void()
    writer = DistributedWriter.get_instance()
    last_train_show_iter = -1

    def _logger(ctx: "OnlineRLContext"):
        if task.finish:
            writer.close()
        nonlocal last_train_show_iter

        if not np.isinf(ctx.eval_value):
            if record_train_iter:
                writer.add_scalar('basic/eval_episode_return_mean-env_step', ctx.eval_value, ctx.env_step)
                writer.add_scalar('basic/eval_episode_return_mean-train_iter', ctx.eval_value, ctx.train_iter)
            else:
                writer.add_scalar('basic/eval_episode_return_mean', ctx.eval_value, ctx.env_step)
        if ctx.train_output is not None and ctx.train_iter - last_train_show_iter >= train_show_freq:
            last_train_show_iter = ctx.train_iter
            if isinstance(ctx.train_output, List):
                output = ctx.train_output.pop()  # only use latest output
            else:
                output = ctx.train_output
            for k, v in output.items():
                if k in ['priority', 'td_error_priority']:
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
    if task.router.is_active and not task.has_role(task.role.LEARNER):
        return task.void()
    writer = DistributedWriter.get_instance()

    def _logger(ctx: "OfflineRLContext"):
        if task.finish:
            writer.close()
        if not np.isinf(ctx.eval_value):
            writer.add_scalar('basic/eval_episode_return_mean-train_iter', ctx.eval_value, ctx.train_iter)
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
        record_path: str = None,
        cfg: Union[dict, EasyDict] = None,
        metric_list: Optional[List[str]] = None,
        env: Optional[BaseEnvManagerV2] = None,
        model: Optional[torch.nn.Module] = None,
        anonymous: bool = False,
        project_name: str = 'default-project',
        wandb_sweep: bool = False,
) -> Callable:
    '''
    Overview:
        Wandb visualizer to track the experiment.
    Arguments:
        - record_path (:obj:`str`): The path to save the replay of simulation.
        - cfg (:obj:`Union[dict, EasyDict]`): Config, a dict of following settings:
            - gradient_logger: boolean. Whether to track the gradient.
            - plot_logger: boolean. Whether to track the metrics like reward and loss.
            - video_logger: boolean. Whether to upload the rendering video replay.
            - action_logger: boolean. `q_value` or `action probability`.
            - return_logger: boolean. Whether to track the return value.
        - metric_list (:obj:`Optional[List[str]]`): Logged metric list, specialized by different policies.
        - env (:obj:`BaseEnvManagerV2`): Evaluator environment.
        - model (:obj:`nn.Module`): Policy neural network model.
        - anonymous (:obj:`bool`): Open the anonymous mode of wandb or not.
            The anonymous mode allows visualization of data without wandb count.
        - project_name (:obj:`str`): The name of wandb project.
    '''
    if task.router.is_active and not task.has_role(task.role.LEARNER):
        return task.void()
    color_list = ["orange", "red", "blue", "purple", "green", "darkcyan"]
    if metric_list is None:
        metric_list = ["q_value", "target q_value", "loss", "lr", "entropy", "target_q_value", "td_error"]
    # Initialize wandb with default settings
    # Settings can be covered by calling wandb.init() at the top of the script
    if not wandb_sweep:
        if anonymous:
            wandb.init(project=project_name, reinit=True, anonymous="must")
        else:
            wandb.init(project=project_name, reinit=True)
    else:
        if anonymous:
            wandb.init(project=project_name, anonymous="must")
        else:
            wandb.init(project=project_name)
        plt.switch_backend('agg')
    if cfg is None:
        cfg = EasyDict(
            dict(
                gradient_logger=False,
                plot_logger=True,
                video_logger=False,
                action_logger=False,
                return_logger=False,
            )
        )
    else:
        if not isinstance(cfg, EasyDict):
            cfg = EasyDict(cfg)
        assert set(cfg.keys()
                   ) == set(["gradient_logger", "plot_logger", "video_logger", "action_logger", "return_logger"])
        assert all(value in [True, False] for value in cfg.values())

    # The visualizer is called to save the replay of the simulation
    # which will be uploaded to wandb later
    if env is not None and cfg.video_logger is True and record_path is not None:
        env.enable_save_replay(replay_path=record_path)
    if cfg.gradient_logger:
        wandb.watch(model)
    else:
        one_time_warning(
            "If you want to use wandb to visualize the gradient, please set gradient_logger = True in the config."
        )

    first_plot = True

    def _plot(ctx: "OnlineRLContext"):
        nonlocal first_plot
        if first_plot:
            first_plot = False
            ctx.wandb_url = wandb.run.get_project_url()

        info_for_logging = {}

        if cfg.plot_logger:
            for metric in metric_list:
                if isinstance(ctx.train_output, Dict) and metric in ctx.train_output:
                    if isinstance(ctx.train_output[metric], torch.Tensor):
                        info_for_logging.update({metric: ctx.train_output[metric].cpu().detach().numpy()})
                    else:
                        info_for_logging.update({metric: ctx.train_output[metric]})
                elif isinstance(ctx.train_output, List) and len(ctx.train_output) > 0 and metric in ctx.train_output[0]:
                    metric_value_list = []
                    for item in ctx.train_output:
                        if isinstance(item[metric], torch.Tensor):
                            metric_value_list.append(item[metric].cpu().detach().numpy())
                        else:
                            metric_value_list.append(item[metric])
                    metric_value = np.mean(metric_value_list)
                    info_for_logging.update({metric: metric_value})
        else:
            one_time_warning(
                "If you want to use wandb to visualize the result, please set plot_logger = True in the config."
            )

        if ctx.eval_value != -np.inf:
            info_for_logging.update(
                {
                    "episode return mean": ctx.eval_value,
                    "episode return std": ctx.eval_value_std,
                    "train iter": ctx.train_iter,
                    "env step": ctx.env_step
                }
            )

            eval_output = ctx.eval_output['output']
            episode_return = ctx.eval_output['episode_return']
            episode_return = np.array(episode_return)
            if len(episode_return.shape) == 2:
                episode_return = episode_return.squeeze(1)

            if cfg.video_logger:
                file_list = []
                for p in os.listdir(record_path):
                    if os.path.splitext(p)[-1] == ".mp4":
                        file_list.append(p)
                file_list.sort(key=lambda fn: os.path.getmtime(os.path.join(record_path, fn)))
                video_path = os.path.join(record_path, file_list[-2])
                info_for_logging.update({"video": wandb.Video(video_path, format="mp4")})

            if cfg.action_logger:
                action_path = os.path.join(record_path, (str(ctx.env_step) + "_action.gif"))
                if all(['logit' in v for v in eval_output]) or hasattr(eval_output, "logit"):
                    if isinstance(eval_output, tnp.ndarray):
                        action_prob = softmax(eval_output.logit)
                    else:
                        action_prob = [softmax(to_ndarray(v['logit'])) for v in eval_output]
                    fig, ax = plt.subplots()
                    plt.ylim([-1, 1])
                    action_dim = len(action_prob[1])
                    x_range = [str(x + 1) for x in range(action_dim)]
                    ln = ax.bar(x_range, [0 for x in range(action_dim)], color=color_list[:action_dim])
                    ani = animation.FuncAnimation(
                        fig, action_prob, fargs=(action_prob, ln), blit=True, save_count=len(action_prob)
                    )
                    ani.save(action_path, writer='pillow')
                    info_for_logging.update({"action": wandb.Video(action_path, format="gif")})

                elif all(['action' in v for v in eval_output[0]]):
                    for i, action_trajectory in enumerate(eval_output):
                        fig, ax = plt.subplots()
                        fig_data = np.array([[i + 1, *v['action']] for i, v in enumerate(action_trajectory)])
                        steps = fig_data[:, 0]
                        actions = fig_data[:, 1:]
                        plt.ylim([-1, 1])
                        for j in range(actions.shape[1]):
                            ax.scatter(steps, actions[:, j])
                        info_for_logging.update({"actions_of_trajectory_{}".format(i): fig})

            if cfg.return_logger:
                return_path = os.path.join(record_path, (str(ctx.env_step) + "_return.gif"))
                fig, ax = plt.subplots()
                ax = plt.gca()
                ax.set_ylim([0, 1])
                hist, x_dim = return_distribution(episode_return)
                assert len(hist) == len(x_dim)
                ln_return = ax.bar(x_dim, hist, width=1, color='r', linewidth=0.7)
                ani = animation.FuncAnimation(fig, return_prob, fargs=(hist, ln_return), blit=True, save_count=1)
                ani.save(return_path, writer='pillow')
                info_for_logging.update({"return distribution": wandb.Video(return_path, format="gif")})

        if bool(info_for_logging):
            wandb.log(data=info_for_logging, step=ctx.env_step)
        plt.clf()

    return _plot


def wandb_offline_logger(
        dataset_path: str,
        record_path: str = None,
        cfg: Union[dict, EasyDict] = None,
        metric_list: Optional[List[str]] = None,
        env: Optional[BaseEnvManagerV2] = None,
        model: Optional[torch.nn.Module] = None,
        anonymous: bool = False,
        project_name: str = 'default-project',
        wandb_sweep: bool = False,
) -> Callable:
    '''
    Overview:
        Wandb visualizer to track the experiment.
    Arguments:
        - datasetpath (:obj:`str`): The path to save the replay of simulation.
        - record_path (:obj:`str`): The path to save the replay of simulation.
        - cfg (:obj:`Union[dict, EasyDict]`): Config, a dict of following settings:
            - gradient_logger: boolean. Whether to track the gradient.
            - plot_logger: boolean. Whether to track the metrics like reward and loss.
            - video_logger: boolean. Whether to upload the rendering video replay.
            - action_logger: boolean. `q_value` or `action probability`.
            - return_logger: boolean. Whether to track the return value.
        - metric_list (:obj:`Optional[List[str]]`): Logged metric list, specialized by different policies.
        - env (:obj:`BaseEnvManagerV2`): Evaluator environment.
        - model (:obj:`nn.Module`): Policy neural network model.
        - anonymous (:obj:`bool`): Open the anonymous mode of wandb or not.
            The anonymous mode allows visualization of data without wandb count.
        - project_name (:obj:`str`): The name of wandb project.
    '''
    if task.router.is_active and not task.has_role(task.role.LEARNER):
        return task.void()
    color_list = ["orange", "red", "blue", "purple", "green", "darkcyan"]
    if metric_list is None:
        metric_list = ["q_value", "target q_value", "loss", "lr", "entropy", "target_q_value", "td_error"]
    # Initialize wandb with default settings
    # Settings can be covered by calling wandb.init() at the top of the script
    if not wandb_sweep:
        if anonymous:
            wandb.init(project=project_name, reinit=True, anonymous="must")
        else:
            wandb.init(project=project_name, reinit=True)
    else:
        if anonymous:
            wandb.init(project=project_name, anonymous="must")
        else:
            wandb.init(project=project_name)
        plt.switch_backend('agg')
    if cfg is None:
        cfg = EasyDict(
            dict(
                gradient_logger=False,
                plot_logger=True,
                video_logger=False,
                action_logger=False,
                return_logger=False,
            )
        )
    else:
        if not isinstance(cfg, EasyDict):
            cfg = EasyDict(cfg)
        assert set(cfg.keys()
                   ) == set(["gradient_logger", "plot_logger", "video_logger", "action_logger", "return_logger"])
        assert all(value in [True, False] for value in cfg.values())

    # The visualizer is called to save the replay of the simulation
    # which will be uploaded to wandb later
    if env is not None and cfg.video_logger is True and record_path is not None:
        env.enable_save_replay(replay_path=record_path)
    if cfg.gradient_logger:
        wandb.watch(model)
    else:
        one_time_warning(
            "If you want to use wandb to visualize the gradient, please set gradient_logger = True in the config."
        )

    first_plot = True

    def _vis_dataset(datasetpath: str):
        try:
            from sklearn.manifold import TSNE
        except ImportError:
            import sys
            logging.warning("Please install sklearn first, such as `pip3 install scikit-learn`.")
            sys.exit(1)
        try:
            import h5py
        except ImportError:
            import sys
            logging.warning("Please install h5py first, such as `pip3 install h5py`.")
            sys.exit(1)
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
        _vis_dataset(dataset_path)

    def _plot(ctx: "OnlineRLContext"):
        nonlocal first_plot
        if first_plot:
            first_plot = False
            ctx.wandb_url = wandb.run.get_project_url()

        info_for_logging = {}

        if cfg.plot_logger:
            for metric in metric_list:
                if isinstance(ctx.train_output, Dict) and metric in ctx.train_output:
                    if isinstance(ctx.train_output[metric], torch.Tensor):
                        info_for_logging.update({metric: ctx.train_output[metric].cpu().detach().numpy()})
                    else:
                        info_for_logging.update({metric: ctx.train_output[metric]})
                elif isinstance(ctx.train_output, List) and len(ctx.train_output) > 0 and metric in ctx.train_output[0]:
                    metric_value_list = []
                    for item in ctx.train_output:
                        if isinstance(item[metric], torch.Tensor):
                            metric_value_list.append(item[metric].cpu().detach().numpy())
                        else:
                            metric_value_list.append(item[metric])
                    metric_value = np.mean(metric_value_list)
                    info_for_logging.update({metric: metric_value})
        else:
            one_time_warning(
                "If you want to use wandb to visualize the result, please set plot_logger = True in the config."
            )

        if ctx.eval_value != -np.inf:
            info_for_logging.update(
                {
                    "episode return mean": ctx.eval_value,
                    "episode return std": ctx.eval_value_std,
                    "train iter": ctx.train_iter,
                    "env step": ctx.env_step
                }
            )

            eval_output = ctx.eval_output['output']
            episode_return = ctx.eval_output['episode_return']
            episode_return = np.array(episode_return)
            if len(episode_return.shape) == 2:
                episode_return = episode_return.squeeze(1)

            if cfg.video_logger:
                file_list = []
                for p in os.listdir(record_path):
                    if os.path.splitext(p)[-1] == ".mp4":
                        file_list.append(p)
                file_list.sort(key=lambda fn: os.path.getmtime(os.path.join(record_path, fn)))
                video_path = os.path.join(record_path, file_list[-2])
                info_for_logging.update({"video": wandb.Video(video_path, format="mp4")})

            if cfg.action_logger:
                action_path = os.path.join(record_path, (str(ctx.env_step) + "_action.gif"))
                if all(['logit' in v for v in eval_output]) or hasattr(eval_output, "logit"):
                    if isinstance(eval_output, tnp.ndarray):
                        action_prob = softmax(eval_output.logit)
                    else:
                        action_prob = [softmax(to_ndarray(v['logit'])) for v in eval_output]
                    fig, ax = plt.subplots()
                    plt.ylim([-1, 1])
                    action_dim = len(action_prob[1])
                    x_range = [str(x + 1) for x in range(action_dim)]
                    ln = ax.bar(x_range, [0 for x in range(action_dim)], color=color_list[:action_dim])
                    ani = animation.FuncAnimation(
                        fig, action_prob, fargs=(action_prob, ln), blit=True, save_count=len(action_prob)
                    )
                    ani.save(action_path, writer='pillow')
                    info_for_logging.update({"action": wandb.Video(action_path, format="gif")})

                elif all(['action' in v for v in eval_output[0]]):
                    for i, action_trajectory in enumerate(eval_output):
                        fig, ax = plt.subplots()
                        fig_data = np.array([[i + 1, *v['action']] for i, v in enumerate(action_trajectory)])
                        steps = fig_data[:, 0]
                        actions = fig_data[:, 1:]
                        plt.ylim([-1, 1])
                        for j in range(actions.shape[1]):
                            ax.scatter(steps, actions[:, j])
                        info_for_logging.update({"actions_of_trajectory_{}".format(i): fig})

            if cfg.return_logger:
                return_path = os.path.join(record_path, (str(ctx.env_step) + "_return.gif"))
                fig, ax = plt.subplots()
                ax = plt.gca()
                ax.set_ylim([0, 1])
                hist, x_dim = return_distribution(episode_return)
                assert len(hist) == len(x_dim)
                ln_return = ax.bar(x_dim, hist, width=1, color='r', linewidth=0.7)
                ani = animation.FuncAnimation(fig, return_prob, fargs=(hist, ln_return), blit=True, save_count=1)
                ani.save(return_path, writer='pillow')
                info_for_logging.update({"return distribution": wandb.Video(return_path, format="gif")})

        if bool(info_for_logging):
            wandb.log(data=info_for_logging, step=ctx.env_step)
        plt.clf()

    return _plot
