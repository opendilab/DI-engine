from ding.worker import BaseLearner, SampleSerialCollector, InteractionSerialEvaluator
from ding.model import VAC
from ding.policy import PPOPolicy
from ding.envs import DingEnvWrapper, FinalEvalRewardEnv, BaseEnvManager
from ding.config import compile_config
from ding.utils import set_pkg_seed

from dizoo.dmc2gym.config.dmc2gym_ppo_config import cartpole_balance_ppo_config
from tensorboardX import SummaryWriter
import os
import dmc2gym
from dizoo.dmc2gym.envs.dmc2gym_env import *
from easydict import EasyDict


class InfoWrapper(gym.Wrapper):
    def __init__(self, env, cfg):
        super().__init__(env)
        cfg = EasyDict(cfg)
        self._cfg = cfg
        self._observation_space = dmc2gym_env_info[cfg.domain_name][cfg.task_name]["observation_space"](
            from_pixels=self._cfg["from_pixels"],
            height=self._cfg["height"],
            width=self._cfg["width"],
            channels_first=self._cfg["channels_first"]
        )
        self._action_space = dmc2gym_env_info[cfg.domain_name][cfg.task_name]["action_space"]
        self._reward_space = dmc2gym_env_info[cfg.domain_name][cfg.task_name]["reward_space"](self._cfg["frame_skip"])

    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        self._seed = seed
        self._dynamic_seed = dynamic_seed
        np.random.seed(self._seed)

    def _process_obs(self, obs):
        if self._cfg["from_pixels"]:
            obs = to_ndarray(obs).astype(np.uint8)
        else:
            obs = to_ndarray(obs).astype(np.float32)
        return obs

    def step(self, action):
        action = np.array([action]).astype('float32')
        obs, reward, done, info = self.env.step(action)
        return self._process_obs(obs), reward, done, info

    def reset(self):
        obs = self.env.reset()
        return self._process_obs(obs)


def wrapped_dmc2gym_env(cfg):
    default_cfg = {
            "frame_skip": 3,
            "from_pixels": True,
            "visualize_reward": False,
            "height": 100,
            "width": 100,
            "channels_first": True,
    }
    default_cfg.update(cfg)


    return DingEnvWrapper(
        dmc2gym.make(
            domain_name=default_cfg["domain_name"],
            task_name=default_cfg["task_name"],
            seed=1,
            visualize_reward=default_cfg["visualize_reward"],
            from_pixels=default_cfg["from_pixels"],
            height=default_cfg["height"],
            width=default_cfg["width"],
            frame_skip=default_cfg["frame_skip"]
        )
        ,
        cfg={
            'env_wrapper': [
                lambda env: InfoWrapper(env, default_cfg),
                lambda env: FinalEvalRewardEnv(env),
            ]
        }
    )


def main(cfg, seed=0, max_iterations=int(1e10)):
    cfg = compile_config(
        cfg, BaseEnvManager, PPOPolicy, BaseLearner, SampleSerialCollector, InteractionSerialEvaluator,
        save_cfg=True
    )
    collector_env_num, evaluator_env_num = cfg.env.collector_env_num, cfg.env.evaluator_env_num
    collector_env = BaseEnvManager(env_fn=[lambda: wrapped_dmc2gym_env(cartpole_balance_ppo_config.env) for _ in range(collector_env_num)],
                                   cfg=cfg.env.manager)
    evaluator_env = BaseEnvManager(env_fn=[lambda: wrapped_dmc2gym_env(cartpole_balance_ppo_config.env) for _ in range(evaluator_env_num)],
                                   cfg=cfg.env.manager)
    # collector_env = BaseEnvManager(env_fn=[lambda: DMC2GymEnv(cfg.env) for _ in range(collector_env_num)],
    #                                cfg=cfg.env.manager)
    # evaluator_env = BaseEnvManager(env_fn=[lambda: DMC2GymEnv(cfg.env) for _ in range(evaluator_env_num)],
    #                                cfg=cfg.env.manager)

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

    for _ in range(max_iterations):
        if evaluator.should_eval(learner.train_iter):
            stop, reward = evaluator.eval(learner.save_checkpoint, learner.train_iter, collector.envstep)
            if stop:
                break
        new_data = collector.collect(train_iter=learner.train_iter)
        learner.train(new_data, collector.envstep)


if __name__ == '__main__':
    main(cartpole_balance_ppo_config)
