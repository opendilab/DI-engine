import os
import gym
import torch
from tensorboardX import SummaryWriter
from easydict import EasyDict
from functools import partial

from ding.config import compile_config
from ding.worker import BaseLearner, SampleSerialCollector, InteractionSerialEvaluator, AdvancedReplayBuffer
from ding.envs import BaseEnvManager, DingEnvWrapper
from ding.envs import get_vec_env_setting, create_env_manager
from ding.policy import DQNPolicy
from ding.model import DQN
from ding.utils import set_pkg_seed
from ding.rl_utils import get_epsilon_greedy_fn
from dizoo.atari.config.serial.spaceinvaders.spaceinvaders_dqn_config import main_config, create_config

def main(rl_cfg, seed=0):
    main_cfg, create_cfg =rl_cfg
    cfg = compile_config(
        main_cfg,
        BaseEnvManager,
        DQNPolicy,
        BaseLearner,
        SampleSerialCollector,
        InteractionSerialEvaluator,
        AdvancedReplayBuffer,
        create_cfg=create_cfg,
        save_cfg=True
    )

    create_cfg.policy.type = create_cfg.policy.type + '_command'
    env_fn = None
    cfg = compile_config(cfg, seed=seed, env=env_fn, auto=True, create_cfg=create_cfg, save_cfg=True)
    # Create main components: env, policy
    env_fn, collector_env_cfg, evaluator_env_cfg = get_vec_env_setting(cfg.env)
    evaluator_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in evaluator_env_cfg])

    evaluator_env.enable_save_replay(cfg.env.replay_path)  # switch save replay interface

    # Set random seed for all package and instance
    evaluator_env.seed(seed, dynamic_seed=False)
    set_pkg_seed(seed, use_cuda=cfg.policy.cuda)

    # Set up RL Policy
    model = DQN(**cfg.policy.model)
    policy = DQNPolicy(cfg.policy, model=model)
    policy.eval_mode.load_state_dict(torch.load(cfg.policy.load_path, map_location='cpu'))

    # evaluate
    tb_logger = SummaryWriter(os.path.join('./{}/log/'.format(cfg.exp_name), 'serial'))
    evaluator = InteractionSerialEvaluator(
        cfg.policy.eval.evaluator, evaluator_env, policy.eval_mode, tb_logger, exp_name=cfg.exp_name
    )
    evaluator.eval()


if __name__ == "__main__":
    main(rl_cfg=(main_config, create_config),seed=0)
