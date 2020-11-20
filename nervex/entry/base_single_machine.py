import sys
import copy
import time
import numpy as np
import torch

from nervex.data import PrioritizedBuffer, default_collate, default_decollate
from nervex.rl_utils import epsilon_greedy, Adder
from nervex.torch_utils import to_device
from nervex.utils import default_get
from nervex.worker.learner import LearnerHook


class ActorProducerHook(LearnerHook):

    def __init__(self, runner, position, priority, freq):
        super().__init__(name='actor_producer', position=position, priority=priority)
        self._runner = runner
        self._freq = freq

    def __call__(self, engine):
        if engine.last_iter.val % self._freq == 0:
            # at least update buffer once
            last_push_count = self._runner.buffer.push_count
            self._runner.collect_data()
            while not self._runner.is_buffer_enough(last_push_count):
                self._runner.collect_data()


class ActorUpdateHook(LearnerHook):

    def __init__(self, runner, position, priority, freq):
        super().__init__(name='actor_producer', position=position, priority=priority)
        self._runner = runner
        self._freq = freq

    def __call__(self, engine):
        if engine.last_iter.val % self._freq == 0:
            self._runner.actor_agent.load_state_dict(engine.agent.state_dict())
        self._runner.learner_step_count = engine.last_iter.val


class EvaluateHook(LearnerHook):

    def __init__(self, runner, priority, freq):
        super().__init__(name='evaluate', position='after_iter', priority=priority)
        self._runner = runner
        self._freq = freq

    def __call__(self, engine):
        if engine.last_iter.val % self._freq == 0:
            self._runner.evaluator_agent.load_state_dict(engine.agent.state_dict())
            self._runner.evaluate()


class SingleMachineRunner():

    def __init__(self, cfg):
        self.cfg = cfg
        self.algo_type = self.cfg.common.algo_type
        assert self.algo_type in ['dqn', 'ppo'], self.algo_type
        self.use_cuda = self.cfg.learner.use_cuda
        self.batch_size = self.cfg.learner.batch_size
        self.buffer = PrioritizedBuffer(cfg.learner.data.buffer_length, cfg.learner.data.max_reuse)
        assert cfg.learner.data.buffer_length >= max(
            self.batch_size, self.batch_size * cfg.learner.train_step // cfg.learner.data.max_reuse
        )
        if self.algo_type == 'dqn':
            eps_cfg = cfg.learner.eps
            self.bandit = epsilon_greedy(eps_cfg.start, eps_cfg.end, eps_cfg.decay, eps_cfg.type)

        self._setup_env()
        self._setup_learner()
        self._setup_agent()
        self._setup_get_data()

        self.env_buffer = {i: [] for i in range(self.actor_env.env_num)}
        self.adder = Adder(self.use_cuda)
        self.train_step = cfg.learner.train_step
        self.learner.register_hook(ActorUpdateHook(self, 'before_run', 40, self.train_step))
        self.learner.register_hook(ActorProducerHook(self, 'before_run', 100, self.train_step))
        self.learner.register_hook(ActorUpdateHook(self, 'after_iter', 40, self.train_step))
        self.learner.register_hook(ActorProducerHook(self, 'after_iter', 100, self.train_step))
        self.learner.register_hook(EvaluateHook(self, 100, cfg.evaluator.eval_step))
        self.actor_step_count = 0
        self.learner_step_count = 0

    def _setup_get_data(self):

        def fn(batch_size):
            while True:
                data = self.buffer.sample(batch_size)
                if data is not None:
                    return data
                time.sleep(3)

        self.learner.get_data = fn

    def _setup_learner(self):
        """set self.learner"""
        raise NotImplementedError

    def _setup_agent(self):
        """set self.actor_agent and self.evaluator_agent"""
        raise NotImplementedError

    def _setup_env(self):
        """set self.actor_env and self.evaluate_env"""
        raise NotImplementedError

    def _accumulate_data(self, idx, obs, agent_output, timestep):
        if self.algo_type == 'dqn':
            step = {
                'obs': obs,
                'next_obs': timestep.obs,
                'action': agent_output['action'],
                'reward': timestep.reward,
                'done': timestep.done,
            }
        elif self.algo_type == 'ppo':
            step = {
                'obs': obs,
                'action': agent_output['action'],
                'logit': agent_output['logit'],
                'value': agent_output['value'],
                'reward': timestep.reward,
                'done': timestep.done,
            }
        self.env_buffer[idx].append(step)

    def _pack_trajectory(self, idx):
        if self.algo_type == 'dqn':
            data = self.env_buffer[idx]
        elif self.algo_type == 'ppo':
            data = self.adder.get_gae(
                self.env_buffer[idx],
                last_value=torch.zeros(1),
                gamma=self.cfg.learner.ppo.gamma,
                gae_lambda=self.cfg.learner.ppo.gae_lambda
            )
        self.buffer.extend(data)
        self.env_buffer[idx] = []

    def _get_train_kwargs(self):
        if self.algo_type == 'dqn':
            eps_threshold = self.bandit(self.learner_step_count)
            return {'eps': eps_threshold}
        elif self.algo_type == 'ppo':
            return {}

    def collect_data(self):
        self.actor_env.launch()
        obs_pool = {i: None for i in range(self.actor_env.env_num)}
        act_pool = {i: None for i in range(self.actor_env.env_num)}
        while not self.actor_env.done:
            obs = self.actor_env.next_obs
            for i, o in obs.items():
                obs_pool[i] = copy.deepcopy(o)
            env_id = obs.keys()
            agent_obs = default_collate(list(obs.values()))
            if self.use_cuda:
                agent_obs = to_device(agent_obs, 'cuda')

            train_kwargs = self._get_train_kwargs()
            outputs = self.actor_agent.forward(agent_obs, **train_kwargs)

            if self.use_cuda:
                outputs = to_device(outputs, 'cpu')
            outputs = default_decollate(outputs)
            outputs = {i: o for i, o in zip(env_id, outputs)}
            for i, o in outputs.items():
                act_pool[i] = copy.deepcopy(o)

            timestep = self.actor_env.step({k: o['action'] for k, o in outputs.items()})

            for i, t in timestep.items():
                self._accumulate_data(i, obs_pool[i], act_pool[i], t)
                if t.done:
                    self._pack_trajectory(i)
            self.actor_step_count += 1
            if self.actor_step_count % self.cfg.actor.print_freq == 0:
                self.learner.info(
                    'actor run step {} with replay buffer size {} with args {}'.format(
                        self.actor_step_count, self.buffer.validlen, train_kwargs
                    )
                )
        self.actor_env.close()

    def evaluate(self):
        self.evaluate_env.seed([int(time.time()) + i for i in range(self.evaluate_env.env_num)])
        self.evaluate_env.launch()
        episode_count = 0
        rewards = []
        obs_pool = {i: None for i in range(self.evaluate_env.env_num)}
        act_pool = {i: None for i in range(self.evaluate_env.env_num)}
        cum_rewards = [0 for _ in range(self.evaluate_env.env_num)]
        while not self.evaluate_env.done:
            obs = self.evaluate_env.next_obs
            for i, o in obs.items():
                obs_pool[i] = copy.deepcopy(o)
            env_id = obs.keys()
            agent_obs = default_collate(list(obs.values()))
            if self.use_cuda:
                agent_obs = to_device(agent_obs, 'cuda')

            outputs = self.evaluator_agent.forward(agent_obs)

            if self.use_cuda:
                outputs = to_device(outputs, 'cpu')
            outputs = default_decollate(outputs)
            outputs = {i: o for i, o in zip(env_id, outputs)}
            for i, o in outputs.items():
                act_pool[i] = copy.deepcopy(o)

            timestep = self.evaluate_env.step({k: o['action'] for k, o in outputs.items()})

            for i, t in timestep.items():
                cum_rewards[i] += default_get(
                    t.info, 'eval_reward', default_fn=lambda: t.reward.item(), judge_fn=np.isscalar
                )
                if t.done:
                    episode_count += 1
                    rewards.append(copy.deepcopy(cum_rewards[i]))
                    cum_rewards[i] = 0.

        self.evaluate_env.close()
        avg_reward = sum(rewards) / len(rewards)
        self.learner.info('evaluate average reward: {:.3f}\t{}'.format(avg_reward, rewards))
        if avg_reward >= self.cfg.evaluator.stop_val:
            sys.exit(0)

    def run(self):
        self.learner.run()

    def is_buffer_enough(self, last_push_count):
        bs = self.cfg.learner.batch_size
        size = int(1.2 * bs * self.train_step) // (self.cfg.learner.data.max_reuse)
        return self.buffer.push_count - last_push_count >= size and self.buffer.validlen >= 2 * bs
