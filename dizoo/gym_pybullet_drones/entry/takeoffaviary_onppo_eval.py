import os
import gym
import torch
from tensorboardX import SummaryWriter
from easydict import EasyDict

from ding.config import compile_config
from ding.worker import BaseLearner, SampleSerialCollector, InteractionSerialEvaluator, NaiveReplayBuffer
from ding.envs import BaseEnvManager, DingEnvWrapper
from ding.policy import PPOPolicy
from ding.model import VAC
from ding.utils import set_pkg_seed

from dizoo.gym_pybullet_drones.envs.gym_pybullet_drones_env import GymPybulletDronesEnv
from dizoo.gym_pybullet_drones.config.takeoffaviary_onppo_config import takeoffaviary_ppo_config


def main(cfg, seed=0, max_iterations=int(1e10)):
    cfg = compile_config(
        cfg,
        BaseEnvManager,
        PPOPolicy,
        BaseLearner,
        SampleSerialCollector,
        InteractionSerialEvaluator,
        NaiveReplayBuffer,
        save_cfg=True
    )
    collector_env_num, evaluator_env_num = cfg.env.collector_env_num, cfg.env.evaluator_env_num

    cfg.env['record'] = True
    cfg.env['gui'] = True
    cfg.env['print_debug_info'] = True
    cfg.env['plot_observation'] = True

    evaluator_env = BaseEnvManager(
        env_fn=[lambda: GymPybulletDronesEnv(cfg.env) for _ in range(evaluator_env_num)], cfg=cfg.env.manager
    )

    evaluator_env.seed(seed, dynamic_seed=False)
    set_pkg_seed(seed, use_cuda=cfg.policy.cuda)

    model = VAC(**cfg.policy.model)
    policy = PPOPolicy(cfg.policy, model=model)
    policy.eval_mode.load_state_dict(torch.load(cfg.policy.load_path, map_location='cpu'))

    tb_logger = SummaryWriter(os.path.join('./log/', 'serial'))
    evaluator = InteractionSerialEvaluator(cfg.policy.eval.evaluator, evaluator_env, policy.eval_mode, tb_logger)
    evaluator.eval()


if __name__ == "__main__":
    main(takeoffaviary_ppo_config)
