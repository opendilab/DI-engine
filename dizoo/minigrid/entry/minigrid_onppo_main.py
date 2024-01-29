import gym
from ding.worker import BaseLearner, SampleSerialCollector, InteractionSerialEvaluator
from ding.model import VAC
from ding.policy import PPOPolicy
from ding.envs import DingEnvWrapper, EvalEpisodeReturnWrapper, BaseEnvManager
from ding.config import compile_config
from ding.utils import set_pkg_seed

from dizoo.minigrid.config.minigrid_onppo_config import minigrid_ppo_config
from minigrid.wrappers import FlatObsWrapper
import numpy as np
from tensorboardX import SummaryWriter
import os
import gymnasium


class MinigridWrapper(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)
        self._observation_space = gym.spaces.Box(low=float("-inf"), high=float("inf"), shape=(8, ), dtype=np.float32)
        self._action_space = gym.spaces.Discrete(9)
        self._action_space.seed(0)  # default seed
        self.reward_range = (float('-inf'), float('inf'))
        self.max_steps = minigrid_ppo_config.env.max_step

    def step(self, action):
        obs, reward, done, _, info = self.env.step(action)
        self.cur_step += 1
        if self.cur_step > self.max_steps:
            done = True
        return obs, reward, done, info

    def reset(self):
        self.cur_step = 0
        return self.env.reset()[0]


def wrapped_minigrid_env():
    return DingEnvWrapper(
        gymnasium.make(minigrid_ppo_config.env.env_id),
        cfg={
            'env_wrapper': [
                lambda env: FlatObsWrapper(env),
                lambda env: MinigridWrapper(env),
                lambda env: EvalEpisodeReturnWrapper(env),
            ]
        }
    )


def main(cfg, seed=0, max_env_step=int(1e10), max_train_iter=int(1e10)):
    cfg = compile_config(
        cfg, BaseEnvManager, PPOPolicy, BaseLearner, SampleSerialCollector, InteractionSerialEvaluator, save_cfg=True
    )
    collector_env_num, evaluator_env_num = cfg.env.collector_env_num, cfg.env.evaluator_env_num
    collector_env = BaseEnvManager(env_fn=[wrapped_minigrid_env for _ in range(collector_env_num)], cfg=cfg.env.manager)
    evaluator_env = BaseEnvManager(env_fn=[wrapped_minigrid_env for _ in range(evaluator_env_num)], cfg=cfg.env.manager)

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
    main(minigrid_ppo_config)
