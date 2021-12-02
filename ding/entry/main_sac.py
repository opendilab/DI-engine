"""
Main entry
"""
from collections import deque
import threading
from types import GeneratorType
import torch
import numpy as np
import time
from rich import print
from functools import partial
from ding.model import QAC, DQN
from ding.utils import set_pkg_seed
from ding.envs import DingEnvWrapper, BaseEnvManager, get_vec_env_setting
from ding.config import compile_config
from ding.policy import SACPolicy, DQNPolicy
from ding.torch_utils import to_ndarray, to_tensor
from ding.rl_utils import get_epsilon_greedy_fn
from ding.worker.collector.base_serial_evaluator import VectorEvalMonitor
from ding.framework import Task, Parallel
# from dizoo.classic_control.pendulum.config.pendulum_sac_config import main_config, create_config
from dizoo.atari.config.serial.pong.pong_dqn_config import main_config, create_config


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


class Differential:
    """
    Sync train/collect/evaluate speed
    """

    def __init__(self, buffer, collect_env) -> None:
        pass

    def __call__(self, ctx) -> None:
        pass


class Pipeline:

    def __init__(self, cfg, model: torch.nn.Module):
        self.cfg = cfg
        self.model = model
        # self.policy = SACPolicy(cfg.policy, model=model)
        self.policy = DQNPolicy(cfg.policy, model=model)
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

        def on_sync_parallel_ctx(ctx):
            if "collect_transitions" in ctx:
                for t in ctx.collect_transitions:
                    buffer_.push(t)

        task.on("sync_parallel_ctx", on_sync_parallel_ctx)

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


def sample_profiler(buffer, print_per_step=1):
    start_time = None
    start_counter = 0
    start_step = 0
    records = deque(maxlen=print_per_step * 50)
    step_records = deque(maxlen=print_per_step * 50)
    max_mean = 0

    def _sample_profiler(ctx):
        nonlocal start_time, start_counter, start_step, max_mean
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
            max_mean = max(np.mean(records), max_mean)
            print(
                "        Samples/s: {:.2f}, Mean: {:.2f}, Max: {:.2f}, Total: {:.0f};\
 Steps/s: {:.2f}, Mean: {:.2f}, Total: {:.0f}".format(
                    record, np.mean(records), max_mean, end_counter, step_record, np.mean(step_records), ctx.total_step
                )
            )
            start_time, start_counter, start_step = end_time, end_counter, end_step

    return _sample_profiler


def step_profiler(step_name, silent=False, print_per_step=1):
    records = deque(maxlen=print_per_step * 5)

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
            if not silent and ctx.total_step % print_per_step == 0:
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

    model = DQN(**cfg.policy.model)
    # model.share_memory()
    replay_buffer = DequeBuffer()
    sac = Pipeline(cfg, model)

    start = time.time()
    with Task(async_mode=False) as task:
        task.use(sample_profiler(replay_buffer, print_per_step=100))
        task.use(
            step_profiler("evaluate", silent=False, print_per_step=100)(sac.evaluate(evaluator_env)),
            # filter_node=lambda node_id: node_id % 2 == 1
        )
        task.use(
            step_profiler("collect", silent=False, print_per_step=100)(
                task.sequence(sac.act(collector_env), sac.collect(collector_env, replay_buffer, task=task))
            )
        )
        task.use(
            step_profiler("learn", silent=False, print_per_step=100)(sac.learn(replay_buffer, task=task)),
            # filter_node=lambda node_id: node_id % 8 == 0
        )

        print(task.middleware)
        task.run(max_step=100)
    time.sleep(1)
    print("Threads", threading.enumerate())
    print("Total time cost: {:.2f}s".format(time.time() - start))


if __name__ == "__main__":
    # from ding.utils import profiler
    # profiler()
    main(main_config, create_config)
    # Parallel.runner(n_parallel_workers=2)(main, main_config, create_config)
    print("Parent Threads", threading.enumerate())
