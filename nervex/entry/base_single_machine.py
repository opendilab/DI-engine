import sys
import copy
import time
import numpy as np
import torch

from nervex.data import PrioritizedBuffer, default_collate, default_decollate
from nervex.rl_utils import epsilon_greedy, Adder
from nervex.torch_utils import to_device
from nervex.utils import default_get, lists_to_dicts, list_split
from nervex.worker.learner import LearnerHook


class ActorProducerHook(LearnerHook):

    def __init__(self, runner, position, priority, freq):
        super().__init__(name='actor_producer', position=position, priority=priority)
        self._runner = runner
        self._freq = freq

    def __call__(self, engine):
        if engine.last_iter.val % self._freq == 0:
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


class SingleMachineRunner(object):

    def __init__(self, cfg):
        self.cfg = cfg
        self.algo_type = self.cfg.common.algo_type
        assert self.algo_type in ['dqn', 'ppo', 'drqn', 'ddpg', 'qmix'], self.algo_type
        self.use_cuda = self.cfg.learner.use_cuda
        self.buffer = PrioritizedBuffer(cfg.learner.data.buffer_length, cfg.learner.data.max_reuse)
        self.batch_size = cfg.learner.data.batch_size
        self.sample_ratio = cfg.learner.data.get('sample_ratio', 2)
        assert cfg.learner.data.buffer_length >= max(
            self.batch_size, self.batch_size * cfg.learner.train_step // cfg.learner.data.max_reuse
        )
        if self.algo_type in ['dqn', 'drqn', 'qmix']:
            eps_cfg = cfg.learner.eps
            self.bandit = epsilon_greedy(eps_cfg.start, eps_cfg.end, eps_cfg.decay, eps_cfg.type)

        self._setup_env()
        self._setup_learner()
        self._setup_get_data()
        self.learner.launch()
        self._setup_agent()

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
        self.actor_obs_pool = {i: None for i in range(self.actor_env.env_num)}
        self.actor_out_pool = {i: None for i in range(self.actor_env.env_num)}

    def _setup_get_data(self):

        def fn(batch_size):
            while True:
                data = self.buffer.sample(batch_size)
                if data is not None:
                    return data
                time.sleep(3)

        self.learner.get_data = fn

    def _setup_learner(self):
        """setup self.learner"""
        raise NotImplementedError

    def _setup_agent(self):
        """setup self.actor_agent and self.evaluator_agent"""
        raise NotImplementedError

    def _setup_env(self):
        """setup self.actor_env and self.evaluate_env"""
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
        elif self.algo_type in ['drqn']:
            step = {
                'obs': obs,
                'action': agent_output['action'],
                'prev_state': agent_output['prev_state'],
                'reward': timestep.reward,
                'done': timestep.done,
            }
        elif self.algo_type == 'ddpg':
            step = {
                'obs': obs,
                'action': agent_output['action'],
                'next_obs': timestep.obs,
                'reward': timestep.reward,
                'done': timestep.done,
            }
        elif self.algo_type in ['qmix']:
            step = {
                'obs': obs,
                'next_obs': timestep.obs,
                'prev_state': agent_output['prev_state'],
                'action': agent_output['action'],
                'reward': timestep.reward,
                'done': timestep.done,
            }

        self.env_buffer[idx].append(step)

    def _pack_trajectory(self, idx):
        if self.algo_type in ['dqn', 'drqn', 'qmix', 'ddpg']:
            data = self.env_buffer[idx]
        elif self.algo_type == 'ppo':
            data = self.adder.get_gae(
                self.env_buffer[idx],
                last_value=torch.zeros(1),
                gamma=self.cfg.learner.ppo.gamma,
                gae_lambda=self.cfg.learner.ppo.gae_lambda
            )
        if self.algo_type in ['dqn', 'ppo', 'ddpg']:
            self.buffer.extend(data)
        elif self.algo_type in ['drqn']:
            nstep = self.cfg.learner.dqn.nstep
            burnin_step = self.cfg.learner.dqn.burnin_step
            traj_step = nstep * 2 + burnin_step
            data = list_split(data, step=traj_step)
            for d in data:
                self.buffer.append(lists_to_dicts(d))
        elif self.algo_type in ['qmix']:
            data = list_split(data, self.cfg.learner.qmix.traj_step)
            for d in data:
                self.buffer.append(lists_to_dicts(d, recursive=True))
        self.env_buffer[idx] = []

    def _get_train_kwargs(self, env_id):
        if self.algo_type == 'dqn':
            eps_threshold = self.bandit(self.learner_step_count)
            return {'eps': eps_threshold}
        elif self.algo_type in ['drqn', 'qmix']:
            eps_threshold = self.bandit(self.learner_step_count)
            return {'eps': eps_threshold, 'state_id': list(env_id)}
        elif self.algo_type == 'ppo':
            return {}
        elif self.algo_type == 'ddpg':
            return {
                'param': {
                    'mode': 'compute_action'
                },
            }

    def collect_data(self):
        if self.actor_step_count == 0:
            self.actor_agent.reset()
        last_push_count = self.buffer.push_count
        while not self.is_buffer_enough(last_push_count):
            obs = self.actor_env.next_obs
            for i, o in obs.items():
                self.actor_obs_pool[i] = copy.deepcopy(o)
            env_id = obs.keys()
            agent_obs = default_collate(list(obs.values()))
            if self.use_cuda:
                agent_obs = to_device(agent_obs, 'cuda')

            train_kwargs = self._get_train_kwargs(env_id)
            outputs = self.actor_agent.forward({'obs': agent_obs}, **train_kwargs)

            if self.use_cuda:
                outputs = to_device(outputs, 'cpu')
            outputs = default_decollate(outputs)
            outputs = {i: o for i, o in zip(env_id, outputs)}
            for i, o in outputs.items():
                self.actor_out_pool[i] = copy.deepcopy(o)

            timestep = self.actor_env.step({k: o['action'] for k, o in outputs.items()})

            for i, t in timestep.items():
                self._accumulate_data(i, self.actor_obs_pool[i], self.actor_out_pool[i], t)
                if t.done:
                    if self.algo_type in ['drqn', 'qmix']:
                        self.actor_agent.reset(state_id=[i])
                    else:
                        self.actor_agent.reset()
                    self._pack_trajectory(i)
                else:
                    if self.algo_type != 'drqn':
                        bs = self.cfg.learner.data.batch_size
                        size = bs * self.sample_ratio * self.train_step / self.cfg.actor.env_num
                        if len(self.env_buffer[i]) > size:
                            self._pack_trajectory(i)
            self.actor_step_count += 1
            if self.actor_step_count % self.cfg.actor.print_freq == 0:
                self.learner.info(
                    'actor run step {} with replay buffer size {} with args {}'.format(
                        self.actor_step_count, self.buffer.validlen, train_kwargs
                    )
                )

    def evaluate(self):
        self.evaluate_env.seed([int(time.time()) + i for i in range(self.evaluate_env.env_num)])
        self.evaluate_env.reset()
        episode_count = 0
        rewards = []
        obs_pool = {i: None for i in range(self.evaluate_env.env_num)}
        act_pool = {i: None for i in range(self.evaluate_env.env_num)}
        cum_rewards = [0 for _ in range(self.evaluate_env.env_num)]
        while episode_count < self.cfg.evaluator.total_episode_num:
            obs = self.evaluate_env.next_obs
            for i, o in obs.items():
                obs_pool[i] = copy.deepcopy(o)
            env_id = obs.keys()
            agent_obs = default_collate(list(obs.values()))
            if self.use_cuda:
                agent_obs = to_device(agent_obs, 'cuda')

            forward_kwargs = {}
            if self.algo_type in ['drqn', 'qmix']:
                forward_kwargs['state_id'] = list(env_id)
            elif self.algo_type == 'ddpg':
                forward_kwargs['param'] = {'mode': 'compute_action'}
            outputs = self.evaluator_agent.forward({'obs': agent_obs}, **forward_kwargs)

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
                    if self.algo_type in ['drqn', 'qmix']:
                        self.evaluator_agent.reset(state_id=[i])
                    else:
                        self.evaluator_agent.reset()
                    episode_count += 1
                    rewards.append(copy.deepcopy(cum_rewards[i]))
                    cum_rewards[i] = 0.

        avg_reward = sum(rewards) / len(rewards)
        self.learner.info('evaluate average reward: {:.3f}\t{}'.format(avg_reward, rewards))
        if avg_reward >= self.cfg.evaluator.stop_val:
            sys.exit(0)

    def run(self):
        self.learner.run()

    def __del__(self):
        self.actor_env.close()
        self.evaluate_env.close()

    def is_buffer_enough(self, last_push_count):
        bs = self.cfg.learner.data.batch_size
        size = int(self.sample_ratio * bs * self.train_step)
        return self.buffer.push_count - last_push_count >= size and self.buffer.validlen >= bs
