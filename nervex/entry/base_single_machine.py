import sys
import copy
import time

from nervex.data import PrioritizedBuffer, default_collate, default_decollate
from nervex.rl_utils import epsilon_greedy
from nervex.torch_utils import to_device
from nervex.worker.learner import LearnerHook


class ActorProducerHook(LearnerHook):

    def __init__(self, runner, position, priority, freq):
        super().__init__(name='actor_producer', position=position, priority=priority)
        self._runner = runner
        self._freq = freq

    def __call__(self, engine):
        if engine.last_iter.val % self._freq == 0:
            # at least update buffer once
            self._runner.collect_data()
            while not self._runner.is_buffer_enough():
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
            self._runner.evaluate_agent.load_state_dict(engine.agent.state_dict())
            self._runner.evaluate()


class SingleMachineRunner():

    def __init__(self, cfg):
        self.cfg = cfg
        self.use_cuda = self.cfg.learner.use_cuda
        self.batch_size = self.cfg.learner.batch_size
        self._setup_env()
        self.buffer = PrioritizedBuffer(cfg.learner.data.buffer_length, cfg.learner.data.max_reuse)
        self.bandit = epsilon_greedy(0.95, 0.05, 100000)

        self._setup_learner()
        self._setup_agent()
        self._setup_get_data()
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
        """set self.actor_agent and self.evaluate_agent"""
        raise NotImplementedError

    def _setup_env(self):
        """set self.actor_env and self.evaluate_env"""
        raise NotImplementedError

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

            eps_threshold = self.bandit(self.learner_step_count)
            outputs = self.actor_agent.forward(agent_obs, eps=eps_threshold)

            if self.use_cuda:
                outputs = to_device(outputs, 'cpu')
            outputs = default_decollate(outputs)
            outputs = {i: o for i, o in zip(env_id, outputs)}
            for i, o in outputs.items():
                act_pool[i] = copy.deepcopy(o)

            timestep = self.actor_env.step({k: o['action'] for k, o in outputs.items()})

            for i, t in timestep.items():
                step = {
                    'obs': obs_pool[i],
                    'action': act_pool[i]['action'],
                    'next_obs': t.obs,
                    'reward': t.reward,
                    'done': t.done,
                }
                self.buffer.append(step)
            self.actor_step_count += 1
            if self.actor_step_count % 200 == 0:
                self.learner.info(
                    'actor run step {} with replay buffer size {} with eps {:.4f}'.format(
                        self.actor_step_count, self.buffer.validlen, eps_threshold
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

            outputs = self.evaluate_agent.forward(agent_obs)

            if self.use_cuda:
                outputs = to_device(outputs, 'cpu')
            outputs = default_decollate(outputs)
            outputs = {i: o for i, o in zip(env_id, outputs)}
            for i, o in outputs.items():
                act_pool[i] = copy.deepcopy(o)

            timestep = self.evaluate_env.step({k: o['action'] for k, o in outputs.items()})

            for i, t in timestep.items():
                cum_rewards[i] += self.learner.computation_graph.get_weighted_reward(t.reward).item()
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

    def is_buffer_enough(self):
        bs = self.cfg.learner.batch_size
        size = int(1.2 * bs * self.train_step) // (self.cfg.learner.data.max_reuse)
        return self.buffer.validlen >= size and self.buffer.validlen >= 2 * bs
