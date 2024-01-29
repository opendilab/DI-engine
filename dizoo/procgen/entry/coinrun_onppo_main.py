import os
from functools import partial

import gym
import numpy as np
from easydict import EasyDict
from tensorboardX import SummaryWriter

from ding.torch_utils import to_ndarray
from ding.worker import BaseLearner, SampleSerialCollector, InteractionSerialEvaluator
from ding.model import VAC
from ding.policy import PPOPolicy
from ding.envs import DingEnvWrapper, EvalEpisodeReturnWrapper, BaseEnvManager
from ding.config import compile_config
from ding.utils import set_pkg_seed
from dizoo.procgen.config.coinrun_ppo_config import coinrun_ppo_config


class CoinrunWrapper(gym.Wrapper):

    def __init__(self, env, cfg):
        super().__init__(env)
        cfg = EasyDict(cfg)
        self._cfg = cfg
        self._observation_space = gym.spaces.Box(
            low=np.zeros(shape=(3, 64, 64)), high=np.ones(shape=(3, 64, 64)) * 255, shape=(3, 64, 64), dtype=np.float32
        )
        self._action_space = gym.spaces.Discrete(15)
        self._reward_space = gym.spaces.Box(low=float("-inf"), high=float("inf"), shape=(1, ), dtype=np.float32)

    def _process_obs(self, obs):
        obs = to_ndarray(obs)
        obs = np.transpose(obs, (2, 0, 1))
        obs = obs.astype(np.float32)
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self._process_obs(obs), reward, bool(done), info

    def reset(self):
        obs = self.env.reset()
        return self._process_obs(obs)


def wrapped_procgen_env(cfg):
    default_cfg = dict(
        control_level=True,
        start_level=0,
        num_levels=0,
        env_id='coinrun',
    )
    default_cfg.update(cfg)
    default_cfg = EasyDict(default_cfg)

    return DingEnvWrapper(
        gym.make(
            'procgen:procgen-' + default_cfg.env_id + '-v0',
            start_level=default_cfg.start_level,
            num_levels=default_cfg.num_levels
        ) if default_cfg.control_level else
        gym.make('procgen:procgen-' + default_cfg.env_id + '-v0', start_level=0, num_levels=1),
        cfg={
            'env_wrapper': [
                lambda env: CoinrunWrapper(env, default_cfg),
                lambda env: EvalEpisodeReturnWrapper(env),
            ]
        }
    )


def main(cfg, seed=0, max_env_step=int(1e10), max_train_iter=int(1e10)):
    cfg = compile_config(
        cfg, BaseEnvManager, PPOPolicy, BaseLearner, SampleSerialCollector, InteractionSerialEvaluator, save_cfg=True
    )
    collector_env_num, evaluator_env_num = cfg.env.collector_env_num, cfg.env.evaluator_env_num
    collector_env = BaseEnvManager(
        env_fn=[partial(wrapped_procgen_env, cfg=coinrun_ppo_config.env) for _ in range(collector_env_num)],
        cfg=cfg.env.manager
    )
    evaluator_env = BaseEnvManager(
        env_fn=[partial(wrapped_procgen_env, cfg=coinrun_ppo_config.env) for _ in range(evaluator_env_num)],
        cfg=cfg.env.manager
    )

    collector_env.seed(seed)
    evaluator_env.seed(seed, dynamic_seed=False)
    set_pkg_seed(seed, use_cuda=cfg.policy.cuda)

    model = VAC(**cfg.policy.model)
    policy = PPOPolicy(cfg.policy, model=model)
    tb_logger = SummaryWriter(os.path.join('./{}/log/'.format(cfg.exp_name), 'serial'))
    learner = BaseLearner(cfg.policy.learn.learner, policy.learn_mode, tb_logger, exp_name=cfg.exp_name)
    collector = SampleSerialCollector(
        cfg.policy.collect.collector, collector_env, policy.collect_mode, tb_logger, exp_name=cfg.exp_name
    )
    evaluator = InteractionSerialEvaluator(
        cfg.policy.eval.evaluator, evaluator_env, policy.eval_mode, tb_logger, exp_name=cfg.exp_name
    )

    while True:
        if evaluator.should_eval(learner.train_iter):
            stop, reward = evaluator.eval(learner.save_checkpoint, learner.train_iter, collector.envstep)
            if stop:
                break
        new_data = collector.collect(train_iter=learner.train_iter)
        learner.train(new_data, collector.envstep)
        if collector.envstep >= max_env_step or learner.train_iter >= max_train_iter:
            break


if __name__ == '__main__':
    main(coinrun_ppo_config)
