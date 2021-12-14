"""
Main entry
"""
from collections import deque
import torch
import numpy as np
import time
from rich import print
from functools import partial
from ding.model import QAC, DQN
from ding.utils import set_pkg_seed
from ding.envs import DingEnvWrapper, BaseEnvManager, get_vec_env_setting, SyncSubprocessEnvManager
from ding.config import compile_config
from ding.policy import SACPolicy, DQNPolicy
from ding.torch_utils import to_ndarray, to_tensor
from ding.rl_utils import get_epsilon_greedy_fn
from ding.worker.collector.base_serial_evaluator import VectorEvalMonitor
from ding.framework import Task, Parallel
from ding.framework.wrapper import StepTimer
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


def differential(task, model, buffer: DequeBuffer, wait_n_sample: int, collect_before_wait: int):
    """
    Sync train/collect/evaluate speed
    """

    cum_buffer_count = 0
    receive_new_model = False
    collected_before_wait = 0
    finish = False

    def sync_parallel_ctx(ctx):
        nonlocal cum_buffer_count, receive_new_model, finish
        if ctx.finish:
            finish = ctx.finish
        if "collect_transitions" in ctx:
            for transition in ctx.collect_transitions:
                buffer.push(transition)
                cum_buffer_count += 1
        if "model_weight" in ctx:
            model.load_state_dict(ctx.model_weight)
            receive_new_model = True

    task.on("sync_parallel_ctx", sync_parallel_ctx)

    def _differential(ctx):
        print("Model", task.router.node_id, model.state_dict()['head.V.1.0.weight'][0][:10])
        nonlocal cum_buffer_count, receive_new_model, collected_before_wait, finish
        timeout = 5
        if finish:
            ctx.finish = finish
        if "train_iter" in ctx:  # On learner
            # Wait until buffer has enough data
            for _ in range(timeout * 100):
                if cum_buffer_count < wait_n_sample:
                    time.sleep(0.01)
                else:
                    break
            cum_buffer_count = 0
        elif "collect_env_step" in ctx:  # On collector
            collected_before_wait += len(ctx.get("collect_transitions") or [])
            # Wait until receive new model
            for _ in range(timeout * 100):
                if collected_before_wait > collect_before_wait and not receive_new_model:
                    time.sleep(0.01)
                else:
                    break
            if collected_before_wait > collect_before_wait:
                collected_before_wait = 0
            if receive_new_model:
                receive_new_model = False

    return _differential


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
                ctx.finish = True

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


class Reporter():

    def __init__(self, task_name, async_mode, parallel_mode, seed):
        # Calculate execution time
        # Save statistics to report.txt
        self.start = time.time()
        self.task_name = task_name
        self.async_mode = async_mode
        self.parallel_mode = parallel_mode
        self.seed = seed
        self.train_iter = 0
        self.collect_env_step = 0

    def record(self):
        duration = time.time() - self.start
        template = "task:{},seed:{},async:{},parallel:{},train_iter:{},env_step:{},duration:{:.2f}"
        with open("./tmp/report.txt", "a+") as f:
            report = template.format(
                self.task_name, self.seed, self.async_mode, self.parallel_mode, self.train_iter, self.collect_env_step,
                self.duration
            )
            print(report)
            f.write(report + "\n")


def main(cfg, model, async_mode, seed=0):
    with Task(async_mode=async_mode) as task:
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

        task.use_step_wrapper(StepTimer(print_per_step=1))
        task.use(sac.evaluate(evaluator_env), filter_labels=["standalone", "node.0"])
        task.use(
            task.sequence(sac.act(collector_env), sac.collect(collector_env, replay_buffer, task=task)),
            filter_labels=["standalone", "node.[1-9]*"]
        )
        task.use(sac.learn(replay_buffer, task=task), filter_labels=["standalone", "node.0"])
        task.use(
            differential(task, model, replay_buffer, wait_n_sample=96, collect_before_wait=48),
            filter_labels=["distributed"]
        )
        task.run(max_step=10000)
        r.train_iter = task.ctx.train_iter
        r.collect_env_step = task.ctx.collect_env_step


if __name__ == "__main__":
    from ding.utils import Profiler
    Profiler().profile()

    import os
    task_name = "Pong/DQN"
    async_mode = False
    parallel_mode = False
    seed = int(os.environ.get("SEED")) if os.environ.get("SEED") else 0

    r = Reporter(task_name, async_mode, parallel_mode, seed)
    cfg = compile_config(main_config, create_cfg=create_config, auto=True)
    model = DQN(**cfg.policy.model)

    if not parallel_mode:
        main(cfg, model, async_mode, seed)
    else:
        n_parallel_workers = 3
        cfg["env"]["collector_env_num"] //= n_parallel_workers - 1
        Parallel.runner(n_parallel_workers=n_parallel_workers, topology="star")(main, cfg, model, async_mode, seed)
    r.record()
