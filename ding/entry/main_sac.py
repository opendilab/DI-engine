"""
Main entry
"""
from collections import deque
from types import GeneratorType
import gym
import torch
import numpy as np
import time
from rich import print
from functools import partial
from ding.model import QAC
from ding.utils import set_pkg_seed
from ding.envs import DingEnvWrapper, BaseEnvManager, get_vec_env_setting
from ding.config import compile_config
from ding.policy import SACPolicy
from ding.torch_utils import to_ndarray, to_tensor
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


class SACPipeline:

    def __init__(self, cfg, model: torch.nn.Module):
        self.cfg = cfg
        self.model = model
        self.policy = SACPolicy(cfg.policy, model=model)

    def act(self, env):

        def _act(ctx):
            ctx.setdefault("collect_env_step", 0)
            ctx.keep("collect_env_step")
            ctx.obs = env.ready_obs
            policy_output = self.policy.collect_mode.forward(ctx.obs)
            ctx.action = to_ndarray({env_id: output['action'] for env_id, output in policy_output.items()})
            ctx.policy_output = policy_output

        return _act

    def collect(self, env, buffer_, task: Task):

        def on_sync_parallel_ctx(ctx):
            if "collect_transitions" in ctx:
                for t in ctx.collect_transitions:
                    buffer_.push(t)

        # task.on("sync_parallel_ctx", on_sync_parallel_ctx)

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

    def learn(self, buffer_, task: Task):

        def _learn(ctx):
            ctx.setdefault("train_iter", 0)
            ctx.keep("train_iter")
            for i in range(self.cfg.policy.learn.update_per_collect):
                data = buffer_.sample(self.policy.learn_mode.get_attribute('batch_size'))
                if data is None:
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
                ctx.finish()

        return _eval


def sample_profiler(buffer, print_per_step=1, async_mode=False):
    start_time = None
    start_counter = 0
    start_step = 0
    records = deque(maxlen=10)
    step_records = deque(maxlen=10)

    def _sample_profiler(ctx):
        nonlocal start_time, start_counter, start_step
        if not start_time:
            start_time = time.time()
        elif ctx.total_step % print_per_step == 0:
            end_time = time.time()
            end_counter = buffer.n_counter
            end_step = ctx.total_step
            record = (end_counter - start_counter) / (end_time - start_time)
            step_record = (end_step - start_step) / (end_time - start_time)
            records.append(record)
            step_records.append(step_record)
            print(
                "        Samples/s: {:.2f}, Mean: {:.2f}, Total: {:.0f}; Steps/s: {:.2f}, Mean: {:.2f}, Total: {:.0f}".
                format(record, np.mean(records), end_counter, step_record, np.mean(step_records), ctx.total_step)
            )
            start_time, start_counter, start_step = end_time, end_counter, end_step

    async def async_sample_profiler(ctx):
        _sample_profiler(ctx)

    return async_sample_profiler if async_mode else _sample_profiler


def step_profiler(step_name, silent=False):
    records = deque(maxlen=10)

    def _step_wrapper(fn):
        # Wrap step function
        def _step_executor(ctx):
            # Execute step
            start_time = time.time()
            time_cost = 0
            g = fn(ctx)
            if isinstance(g, GeneratorType):
                next(g)
                time_cost = time.time() - start_time
                yield
                start_time = time.time()
                try:
                    next(g)
                except StopIteration:
                    pass
                time_cost += time.time() - start_time
            else:
                time_cost = time.time() - start_time
            records.append(time_cost * 1000)
            if not silent:
                print(
                    "    Step Profiler {}: Cost: {:.2f}ms, Mean: {:.2f}ms".format(
                        step_name, time_cost * 1000, np.mean(records)
                    )
                )

        return _step_executor

    return _step_wrapper


def mock_pipeline(buffer):

    def _mock_pipeline(ctx):
        buffer.push(0)

    return _mock_pipeline


def print_step(task: Task):
    import random
    from os import path
    time.sleep(random.random() + 1)
    print(
        "Current task step on {}".format(task.parallel_mode and path.basename(task.router._bind_addr or "")),
        task.ctx.total_step
    )


def main(cfg, create_cfg, seed=0):
    cfg = compile_config(cfg, create_cfg=create_cfg, auto=True)

    env_fn, collector_env_cfg, evaluator_env_cfg = get_vec_env_setting(cfg.env)

    collector_env = BaseEnvManager(env_fn=[partial(env_fn, cfg=c) for c in collector_env_cfg], cfg=cfg.env.manager)
    evaluator_env = BaseEnvManager(env_fn=[partial(env_fn, cfg=c) for c in evaluator_env_cfg], cfg=cfg.env.manager)

    collector_env.seed(seed)
    evaluator_env.seed(seed, dynamic_seed=False)
    set_pkg_seed(seed, use_cuda=cfg.policy.cuda)
    collector_env.launch()
    evaluator_env.launch()

    model = QAC(**cfg.policy.model)
    replay_buffer = DequeBuffer()

    task = Task(async_mode=False, n_async_workers=3, parallel_mode=False, n_parallel_workers=2, attach_to=[])
    sac = SACPipeline(cfg, model)

    task.use(sac.evaluate(evaluator_env))
    task.use(task.sequence(sac.act(collector_env), sac.collect(collector_env, replay_buffer, task=task)))
    task.use(sac.learn(replay_buffer, task=task))

    start = time.time()
    task.run(max_step=10000)
    print("Total time cost: {:.2f}s".format(time.time() - start))


if __name__ == "__main__":
    main(main_config, create_config)
