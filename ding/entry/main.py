"""
Main entry
"""
from collections import deque
import torch
import numpy as np
import time
from rich import print
from functools import partial
from ding.model import QAC
from ding.utils import set_pkg_seed
from ding.envs import BaseEnvManager, get_vec_env_setting
from ding.config import compile_config
from ding.policy import SACPolicy
from ding.torch_utils import to_ndarray, to_tensor
from ding.rl_utils import get_epsilon_greedy_fn
from ding.worker.collector.base_serial_evaluator import VectorEvalMonitor
from ding.framework import Task
from dizoo.classic_control.pendulum.config.pendulum_sac_config import main_config, create_config


class DequeBuffer:
    """
    For demonstration only
    """

    def __init__(self, maxlen=20000) -> None:
        self.memory = deque(maxlen=maxlen)
        self.n_counter = 0

    def push(self, data):
        self.memory.append(data)
        self.n_counter += 1

    def sample(self, size):
        if size > len(self.memory):
            print('[Warning] no enough data: {}/{}'.format(size, len(self.memory)))
            return None
        indices = list(np.random.choice(a=len(self.memory), size=size, replace=False))
        return [self.memory[i] for i in indices]
        # return random.sample(self.memory, size)

    def count(self):
        return len(self.memory)


class Pipeline:

    def __init__(self, cfg, model: torch.nn.Module):
        self.cfg = cfg
        self.model = model
        self.policy = SACPolicy(cfg.policy, model=model)
        if 'eps' in cfg.policy.other:
            eps_cfg = cfg.policy.other.eps
            self.epsilon_greedy = get_epsilon_greedy_fn(eps_cfg.start, eps_cfg.end, eps_cfg.decay, eps_cfg.type)

    def act(self, env):

        def _act(ctx):
            ctx.setdefault("collect_env_step", 0)
            ctx.keep("collect_env_step")
            ctx.obs = env.ready_obs
            policy_kwargs = {}
            if hasattr(self, 'epsilon_greedy'):
                policy_kwargs['eps'] = self.epsilon_greedy(ctx.collect_env_step)
            policy_output = self.policy.collect_mode.forward(ctx.obs, **policy_kwargs)
            ctx.action = to_ndarray({env_id: output['action'] for env_id, output in policy_output.items()})
            ctx.policy_output = policy_output

        return _act

    def collect(self, env, buffer_, task: Task):

        def _collect(ctx):
            timesteps = env.step(ctx.action)
            ctx.collect_env_step += len(timesteps)
            timesteps = to_tensor(timesteps, dtype=torch.float32)
            ctx.collect_transitions = []
            for env_id, timestep in timesteps.items():
                transition = self.policy.collect_mode.process_transition(
                    ctx.obs[env_id], ctx.policy_output[env_id], timestep
                )
                ctx.collect_transitions.append(transition)
                buffer_.push(transition)

        return _collect

    def learn(self, buffer_: DequeBuffer, task: Task):

        def _learn(ctx):
            ctx.setdefault("train_iter", 0)
            ctx.keep("train_iter")
            for i in range(self.cfg.policy.learn.update_per_collect):
                data = buffer_.sample(self.policy.learn_mode.get_attribute('batch_size'))
                if not data:
                    break
                learn_output = self.policy.learn_mode.forward(data)
                if ctx.train_iter % 20 == 0:
                    print(
                        'Current Training: Train Iter({})\tLoss({:.3f})'.format(
                            ctx.train_iter, learn_output['total_loss']
                        )
                    )
                ctx.train_iter += 1

        return _learn

    def evaluate(self, env):

        def _eval(ctx):
            ctx.setdefault("train_iter", 0)
            ctx.setdefault("last_eval_iter", -1)
            ctx.keep("train_iter", "last_eval_iter")
            if ctx.train_iter == ctx.last_eval_iter or (
                (ctx.train_iter - ctx.last_eval_iter) < self.cfg.policy.eval.evaluator.eval_freq
                    and ctx.train_iter != 0):
                return
            env.reset()
            eval_monitor = VectorEvalMonitor(env.env_num, self.cfg.env.n_evaluator_episode)
            while not eval_monitor.is_finished():
                obs = env.ready_obs
                obs = to_tensor(obs, dtype=torch.float32)
                policy_output = self.policy.eval_mode.forward(obs)
                action = to_ndarray({i: a['action'] for i, a in policy_output.items()})
                timesteps = env.step(action)
                timesteps = to_tensor(timesteps, dtype=torch.float32)
                for env_id, timestep in timesteps.items():
                    if timestep.done:
                        self.policy.eval_mode.reset([env_id])
                        reward = timestep.info['final_eval_reward']
                        eval_monitor.update_reward(env_id, reward)
            episode_reward = eval_monitor.get_episode_reward()
            eval_reward = np.mean(episode_reward)
            stop_flag = eval_reward >= self.cfg.env.stop_value and ctx.train_iter > 0
            print('Current Evaluation: Train Iter({})\tEval Reward({:.3f})'.format(ctx.train_iter, eval_reward))
            ctx.last_eval_iter = ctx.train_iter
            if stop_flag:
                ctx.finish = True

        return _eval


def main(cfg, model, seed=0):
    with Task(async_mode=False) as task:
        env_fn, collector_env_cfg, evaluator_env_cfg = get_vec_env_setting(cfg.env)

        collector_env = BaseEnvManager(env_fn=[partial(env_fn, cfg=c) for c in collector_env_cfg], cfg=cfg.env.manager)
        evaluator_env = BaseEnvManager(env_fn=[partial(env_fn, cfg=c) for c in evaluator_env_cfg], cfg=cfg.env.manager)

        collector_env.seed(seed)
        evaluator_env.seed(seed, dynamic_seed=False)
        set_pkg_seed(seed, use_cuda=cfg.policy.cuda)
        collector_env.launch()
        evaluator_env.launch()

        replay_buffer = DequeBuffer()
        sac = Pipeline(cfg, model)

        # task.use_step_wrapper(StepTimer(print_per_step=1))
        task.use(sac.evaluate(evaluator_env), filter_labels=["standalone", "node.0"])
        task.use(
            task.sequence(sac.act(collector_env), sac.collect(collector_env, replay_buffer, task=task)),
            filter_labels=["standalone", "node.[1-9]*"]
        )
        task.use(sac.learn(replay_buffer, task=task), filter_labels=["standalone", "node.0"])
        task.run(max_step=100000)


if __name__ == "__main__":
    cfg = compile_config(main_config, create_cfg=create_config, auto=True)
    model = QAC(**cfg.policy.model)
    main(cfg, model)
