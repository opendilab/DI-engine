"""
Main entry
"""
from collections import deque
import gym
import torch
import numpy as np
import time
from rich import print
from ding.model import DQN
from ding.utils import set_pkg_seed
from ding.envs import DingEnvWrapper, BaseEnvManager
from ding.config import compile_config
from ding.policy import DQNPolicy
from ding.rl_utils import get_epsilon_greedy_fn
from ding.torch_utils import to_ndarray, to_tensor
from ding.worker.collector.base_serial_evaluator import VectorEvalMonitor
from ding.framework import Task
from dizoo.classic_control.cartpole.config.cartpole_dqn_config import main_config, create_config


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


class DQNPipeline:

    def __init__(self, cfg, model):
        self.cfg = cfg
        self.policy = DQNPolicy(cfg.policy, model=model)
        eps_cfg = cfg.policy.other.eps
        self.epsilon_greedy = get_epsilon_greedy_fn(eps_cfg.start, eps_cfg.end, eps_cfg.decay, eps_cfg.type)

    def act(self, env):

        def _act(ctx):
            ctx.setdefault("collect_env_step", 0)
            ctx.keep("collect_env_step")
            eps = self.epsilon_greedy(ctx.collect_env_step)
            ctx.obs = env.ready_obs
            policy_output = self.policy.collect_mode.forward(ctx.obs, eps=eps)
            ctx.action = to_ndarray({env_id: output['action'] for env_id, output in policy_output.items()})
            ctx.policy_output = policy_output
            yield

        return _act

    def collect(self, env, buffer_):

        def _collect(ctx):
            timesteps = env.step(ctx.action)
            ctx.collect_env_step += len(timesteps)
            timesteps = to_tensor(timesteps, dtype=torch.float32)
            for env_id, timestep in timesteps.items():
                transition = self.policy.collect_mode.process_transition(
                    ctx.obs[env_id], ctx.policy_output[env_id], timestep
                )
                buffer_.push(transition)
            yield

        return _collect

    def learn(self, buffer_):

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
            yield

        return _learn

    def evaluate(self, env):

        def _eval(ctx):
            ctx.setdefault("train_iter", 0)
            ctx.setdefault("last_eval_iter", -1)
            ctx.keep("train_iter", "last_eval_iter")
            if ctx.train_iter == ctx.last_eval_iter or (
                (ctx.train_iter - ctx.last_eval_iter) < self.cfg.policy.eval.evaluator.eval_freq
                    and ctx.train_iter != 0):
                yield
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
            yield

        return _eval


def sample_profile(buffer):
    start_time = None
    start_counter = 0
    start_step = 0
    records = deque(maxlen=10)
    step_records = deque(maxlen=10)

    def _sample_profile(ctx):
        nonlocal start_time, start_counter, start_step
        if not start_time:
            start_time = time.time()
        elif ctx.total_step % 30 == 0:
            end_time = time.time()
            end_counter = buffer.n_counter
            end_step = ctx.total_step
            record = (end_counter - start_counter) / (end_time - start_time)
            step_record = (end_step - start_step) / (end_time - start_time)
            records.append(record)
            step_records.append(step_record)
            print(
                ">>>>>>>>> Samples/s: {:.1f}, Mean: {:.1f}, Total: {:.0f}; Steps/s: {:.1f}, Mean: {:.1f}, Total: {:.0f}"
                .format(record, np.mean(records), end_counter, step_record, np.mean(step_records), ctx.total_step)
            )
            start_time, start_counter = end_time, end_counter

    return _sample_profile


def mock_pipeline(buffer):

    def _mock_pipeline(ctx):
        buffer.push(0)

    return _mock_pipeline


def main(cfg, create_cfg, seed=0):

    def wrapped_cartpole_env():
        return DingEnvWrapper(gym.make('CartPole-v0'))

    cfg = compile_config(cfg, create_cfg=create_cfg, auto=True)
    collector_env_num, evaluator_env_num = cfg.env.collector_env_num, cfg.env.evaluator_env_num
    collector_env = BaseEnvManager(env_fn=[wrapped_cartpole_env for _ in range(collector_env_num)], cfg=cfg.env.manager)
    evaluator_env = BaseEnvManager(env_fn=[wrapped_cartpole_env for _ in range(evaluator_env_num)], cfg=cfg.env.manager)

    collector_env.seed(seed)
    evaluator_env.seed(seed, dynamic_seed=False)
    set_pkg_seed(seed, use_cuda=cfg.policy.cuda)
    collector_env.launch()
    evaluator_env.launch()

    model = DQN(**cfg.policy.model)
    replay_buffer = DequeBuffer()

    task = Task()
    dqn = DQNPipeline(cfg, model)

    task.use(sample_profile(replay_buffer))
    # task.use(mock_pipeline(replay_buffer))
    task.use(dqn.evaluate(evaluator_env))
    task.use(dqn.act(collector_env))
    task.use(dqn.collect(collector_env, replay_buffer))
    task.use(dqn.learn(replay_buffer))

    task.run(max_step=10000)


def main_eager(cfg, create_cfg, seed=0):

    def wrapped_cartpole_env():
        return DingEnvWrapper(gym.make('CartPole-v0'))

    cfg = compile_config(cfg, create_cfg=create_cfg, auto=True)
    collector_env_num, evaluator_env_num = cfg.env.collector_env_num, cfg.env.evaluator_env_num
    collector_env = BaseEnvManager(env_fn=[wrapped_cartpole_env for _ in range(collector_env_num)], cfg=cfg.env.manager)
    evaluator_env = BaseEnvManager(env_fn=[wrapped_cartpole_env for _ in range(evaluator_env_num)], cfg=cfg.env.manager)

    collector_env.seed(seed)
    evaluator_env.seed(seed, dynamic_seed=False)
    set_pkg_seed(seed, use_cuda=cfg.policy.cuda)
    collector_env.launch()
    evaluator_env.launch()

    model = DQN(**cfg.policy.model)
    replay_buffer = DequeBuffer()

    task = Task()
    dqn = DQNPipeline(cfg, model)

    evaluate = dqn.evaluate(evaluator_env)
    act = dqn.act(collector_env)
    collect = dqn.collect(collector_env, replay_buffer)
    learn = dqn.learn(replay_buffer)
    profile = sample_profile(replay_buffer)

    for i in range(1000):
        task.forward(profile)
        task.forward(evaluate)
        if task.finish:
            break
        task.forward(act)
        task.forward(collect)
        task.forward(learn)
        task.backward()
        task.renew()


if __name__ == "__main__":
    # main(main_config, create_config)
    main_eager(main_config, create_config)
