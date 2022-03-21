import torch
import numpy as np
from typing import TYPE_CHECKING
from ding.envs.env_manager import BaseEnvManager
from ding.worker.collector.base_serial_evaluator import VectorEvalMonitor
from ding.torch_utils import to_ndarray, to_tensor
from ding.policy import Policy
import logging

if TYPE_CHECKING:
    from ding.framework import Task, Context


def basic_evaluator(task: "Task", cfg: dict, policy: Policy, env: BaseEnvManager):
    env.seed(cfg.seed, dynamic_seed=False)

    def _evaluate(ctx: "Context"):
        if env._closed:
            env.launch()

        ctx.setdefault("train_iter", 0)
        ctx.setdefault("last_eval_iter", -1)
        ctx.keep("train_iter", "last_eval_iter")
        if ctx.train_iter == ctx.last_eval_iter or \
            ((ctx.train_iter - ctx.last_eval_iter) <
                cfg.policy.eval.evaluator.eval_freq and ctx.train_iter != 0):
            return
        env.reset()
        eval_monitor = VectorEvalMonitor(env.env_num, cfg.env.n_evaluator_episode)
        while not eval_monitor.is_finished():
            obs = env.ready_obs
            obs = to_tensor(obs, dtype=torch.float32)
            policy_output = policy.eval_mode.forward(obs)
            action = to_ndarray({i: a['action'] for i, a in policy_output.items()})
            timesteps = env.step(action)
            timesteps = to_tensor(timesteps, dtype=torch.float32)
            for env_id, timestep in timesteps.items():
                if timestep.done:
                    policy.eval_mode.reset([env_id])
                    reward = timestep.info['final_eval_reward']
                    eval_monitor.update_reward(env_id, reward)
        episode_reward = eval_monitor.get_episode_reward()
        eval_reward = np.mean(episode_reward)
        stop_flag = eval_reward >= cfg.env.stop_value and ctx.train_iter > 0
        logging.info('Current Evaluation: Train Iter({})\tEval Reward({:.3f})'.format(ctx.train_iter, eval_reward))
        ctx.last_eval_iter = ctx.train_iter
        if stop_flag:
            task.finish = True

    return _evaluate
