from typing import TYPE_CHECKING, Callable, Dict, List
import numpy as np
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
