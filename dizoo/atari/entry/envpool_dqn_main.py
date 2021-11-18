import os
import gym
from tensorboardX import SummaryWriter
from easydict import EasyDict
from copy import deepcopy
from functools import partial

from ding.config import compile_config
from ding.worker import BaseLearner, SampleSerialCollector, InteractionSerialEvaluator, AdvancedReplayBuffer
from ding.envs import SyncSubprocessEnvManager
from ding.policy import DQNPolicy
from ding.model import DQN
from ding.utils import set_pkg_seed, deep_merge_dicts
from ding.rl_utils import get_epsilon_greedy_fn
from dizoo.atari.envs import AtariEnv
from dizoo.atari.config.serial.spaceinvaders.spaceinvaders_dqn_config import spaceinvaders_dqn_config

import envpool
import numpy as np
from easydict import EasyDict
from copy import deepcopy
from ding.envs import BaseEnvTimestep


class PoolEnvManager:

    @classmethod
    def default_config(cls):
        return EasyDict(deepcopy(cls.config))

    config = dict(type='pool', )

    def __init__(self, cfg):
        self._cfg = cfg
        self._env_num = cfg.env_num
        self._batch_size = cfg.batch_size
        self._ready_obs = None

    @property
    def env_num(self):
        return self._env_num

    @property
    def ready_obs(self):
        return self._ready_obs

    def launch(self):
        self._envs = envpool.make(self._cfg.env_id, env_type="gym", num_envs=self._env_num, batch_size=self._batch_size)
        obs = self._envs.reset()
        obs = obs.astype(np.float32)
        self._ready_obs = {i: o for i, o in enumerate(obs)}

    def reset(self):
        obs = self._envs.reset()
        obs = obs.astype(np.float32)
        self._ready_obs = {i: o for i, o in enumerate(obs)}
        return self._ready_obs

    def step(self, action):
        env_id = np.array(list(action.keys()))
        action = np.array(list(action.values()))
        action = action.squeeze(1)
        self._envs.send(action, env_id)

        obs, rew, done, info = self._envs.recv()
        obs = obs.astype(np.float32)
        rew = rew.astype(np.float32)
        env_id = info['env_id']
        timesteps = {}
        self._ready_obs = {}
        for i in range(len(env_id)):
            d = bool(done[i])
            r = rew[i:i + 1]
            timesteps[env_id[i]] = BaseEnvTimestep(obs[i], r, d, info={'env_id': i, 'final_eval_reward': 0.})
            self._ready_obs[env_id[i]] = obs[i]
        return timesteps

    def close(self):
        pass

    def seed(self, seed, dynamic_seed=False):
        pass


def main(cfg, seed=0, max_iterations=int(1e10)):
    cfg.exp_name = 'atari_dqn_envpool'
    cfg = compile_config(
        cfg,
        PoolEnvManager,
        DQNPolicy,
        BaseLearner,
        SampleSerialCollector,
        InteractionSerialEvaluator,
        AdvancedReplayBuffer,
        save_cfg=True
    )
    cfg.env.env_id = 'Pong-v5'
    cfg.env.collector_batch_size = cfg.env.collector_env_num
    cfg.env.evaluator_batch_size = cfg.env.evaluator_env_num
    collector_env_cfg = EasyDict(
        {
            'env_id': cfg.env.env_id,
            'env_num': cfg.env.collector_env_num,
            'batch_size': cfg.env.collector_batch_size
        }
    )
    collector_env = PoolEnvManager(collector_env_cfg)
    evaluator_env_cfg = EasyDict(
        {
            'env_id': cfg.env.env_id,
            'env_num': cfg.env.evaluator_env_num,
            'batch_size': cfg.env.evaluator_batch_size
        }
    )
    evaluator_env = PoolEnvManager(evaluator_env_cfg)

    collector_env.seed(seed)
    evaluator_env.seed(seed, dynamic_seed=False)
    set_pkg_seed(seed, use_cuda=cfg.policy.cuda)

    model = DQN(**cfg.policy.model)
    policy = DQNPolicy(cfg.policy, model=model)
    tb_logger = SummaryWriter(os.path.join('./{}/log/'.format(cfg.exp_name), 'serial'))
    learner = BaseLearner(cfg.policy.learn.learner, policy.learn_mode, tb_logger, exp_name=cfg.exp_name)
    collector = SampleSerialCollector(
        cfg.policy.collect.collector, collector_env, policy.collect_mode, tb_logger, exp_name=cfg.exp_name
    )
    evaluator = InteractionSerialEvaluator(
        cfg.policy.eval.evaluator, evaluator_env, policy.eval_mode, tb_logger, exp_name=cfg.exp_name
    )
    replay_buffer = AdvancedReplayBuffer(
        cfg.policy.other.replay_buffer, tb_logger, exp_name=cfg.exp_name, instance_name='replay_buffer'
    )
    eps_cfg = cfg.policy.other.eps
    epsilon_greedy = get_epsilon_greedy_fn(eps_cfg.start, eps_cfg.end, eps_cfg.decay, eps_cfg.type)

    while True:
        if evaluator.should_eval(learner.train_iter):
            stop, reward = evaluator.eval(learner.save_checkpoint, learner.train_iter, collector.envstep)
            if stop:
                break
        eps = epsilon_greedy(collector.envstep)
        new_data = collector.collect(train_iter=learner.train_iter, policy_kwargs={'eps': eps})
        replay_buffer.push(new_data, cur_collector_envstep=collector.envstep)
        for i in range(cfg.policy.learn.update_per_collect):
            batch_size = learner.policy.get_attribute('batch_size')
            train_data = replay_buffer.sample(batch_size, learner.train_iter)
            if train_data is not None:
                learner.train(train_data, collector.envstep)


if __name__ == "__main__":
    main(EasyDict(spaceinvaders_dqn_config))
