import os
from venv import create
import gym
import torch
from tensorboardX import SummaryWriter
from easydict import EasyDict

from ding.config import compile_config
from ding.worker import BaseLearner, SampleSerialCollector, InteractionSerialEvaluator, AdvancedReplayBuffer
from ding.envs import SyncSubprocessEnvManager, DingEnvWrapper, BaseEnvManager
from ding.envs.env_wrappers import MaxAndSkipWrapper, WarpFrameWrapper, ScaledFloatFrameWrapper, FrameStackWrapper, \
    FinalEvalRewardEnv
from ding.policy import DQNPolicy, PPOPolicy
from ding.model import DQN, VAC
from ding.utils import set_pkg_seed
from ding.rl_utils import get_epsilon_greedy_fn
from dizoo.beergame.config.beergame_onppo_config import beergame_ppo_config, beergame_ppo_create_config
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from ding.envs import get_vec_env_setting
from functools import partial


def main(cfg, seed=0):
    env_fn = None
    cfg, create_cfg = beergame_ppo_config, beergame_ppo_create_config
    cfg = compile_config(cfg, seed=seed, env=env_fn, auto=True, create_cfg=create_cfg, save_cfg=True)
    collector_env_num, evaluator_env_num = cfg.env.collector_env_num, cfg.env.evaluator_env_num
    # evaluate
    # Set random seed for all package and instance

    # Set up RL Policyz
    env_fn, collector_env_cfg, evaluator_env_cfg = get_vec_env_setting(cfg.env)
    cfg.env.manager.auto_reset = False
    evaluator_env = BaseEnvManager(env_fn=[partial(env_fn, cfg=c) for c in evaluator_env_cfg], cfg=cfg.env.manager)
    evaluator_env.seed(seed, dynamic_seed=False)
    set_pkg_seed(seed, use_cuda=cfg.policy.cuda)
    model = VAC(**cfg.policy.model)
    tb_logger = SummaryWriter(os.path.join('./{}/log/'.format(cfg.exp_name), 'serial'))
    policy = PPOPolicy(cfg.policy, model=model)  # switch save replay interface
    cfg.policy.eval.evaluator.figure_path = './'
    evaluator = InteractionSerialEvaluator(
        cfg.policy.eval.evaluator, evaluator_env, policy.eval_mode, tb_logger, exp_name=cfg.exp_name
    )
    # model.load_state_dict(torch.load('ckpt_best.pth.tar', map_location='cpu')["model"])

    model.load_state_dict(torch.load('model path', map_location='cpu')["model"])
    evaluator.eval(None, -1, -1)


if __name__ == "__main__":
    beergame_ppo_config.exp_name = 'beergame_evaluate'
    main(beergame_ppo_config)