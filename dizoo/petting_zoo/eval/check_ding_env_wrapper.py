import os

import gym
import torch
from easydict import EasyDict
from pettingzoo.mpe import simple_spread_v2
from tensorboardX import SummaryWriter

from ding.config import compile_config
from ding.envs import BaseEnvManager, BaseEnvManagerV2, DingEnvWrapper
from ding.model.template.vac import VAC
from ding.policy import PPOPolicy
from ding.utils import set_pkg_seed
from ding.worker import BaseLearner, SampleSerialCollector, InteractionSerialEvaluator, AdvancedReplayBuffer
from dizoo.petting_zoo.config.ptz_simple_spread_mappo_config import main_config


def main(cfg, seed=0):
    cfg['exp_name'] = 'simple_spread_mappo_eval'
    cfg = compile_config(
        cfg,
        BaseEnvManagerV2,
        PPOPolicy,
        BaseLearner,
        SampleSerialCollector,
        InteractionSerialEvaluator,
        AdvancedReplayBuffer,
        save_cfg=True
    )
    cfg.policy.load_path = 'pettingzoo/ptz_simple_spread_mappo_seed0/ckpt/ckpt_best.pth.tar'

    # =================================================================================
    # NOTE: now DingEnvWrapper is only support single-agent environments
    # Regarding the support of the multi-agent environment, we will develop it later, thank you for your patience.
    # ==================================================================================
    evaluator_env = BaseEnvManager(
        env_fn=[lambda: DingEnvWrapper(simple_spread_v2.parallel_env(N=3, local_ratio=0.5, max_cycles=25, continuous_actions=False)) for _ in range(cfg.env.evaluator_env_num)],
        cfg=cfg.env.manager
    )

    evaluator_env.enable_save_replay(replay_path='./simple_spread_mappo_eval/video')
    evaluator_env.seed(seed, dynamic_seed=False)
    set_pkg_seed(seed, use_cuda=cfg.policy.cuda)

    model = VAC(**cfg.policy.model)
    policy = PPOPolicy(cfg.policy, model=model)
    policy.eval_mode.load_state_dict(torch.load(cfg.policy.load_path, map_location='cuda'))

    tb_logger = SummaryWriter(os.path.join('./{}/log/'.format(cfg.exp_name), 'serial'))
    evaluator = InteractionSerialEvaluator(
        cfg.policy.eval.evaluator, evaluator_env, policy.eval_mode, tb_logger, exp_name=cfg.exp_name
    )
    evaluator.eval()


if __name__ == "__main__":
    main(main_config)
