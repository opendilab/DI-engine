import os

import torch
from tensorboardX import SummaryWriter

from ding.config import compile_config
from ding.envs import BaseEnvManager
from ding.model.template.mavac import MAVAC
from ding.policy import PPOPolicy
from ding.utils import set_pkg_seed
from ding.worker import BaseLearner, SampleSerialCollector, InteractionSerialEvaluator, AdvancedReplayBuffer
from dizoo.petting_zoo.config.ptz_simple_spread_mappo_config import main_config
from dizoo.petting_zoo.envs.petting_zoo_simple_spread_env import PettingZooEnv


def main(cfg, seed=0):
    main_config.env.save_replay = True
    main_config.env.evaluator_env_num = 8
    main_config.env.n_evaluator_episode = 8
    env = PettingZooEnv(main_config.env)
    cfg['exp_name'] = 'simple_spread_mappo_eval'
    cfg = compile_config(
        cfg,
        BaseEnvManager,
        PPOPolicy,
        BaseLearner,
        SampleSerialCollector,
        InteractionSerialEvaluator,
        AdvancedReplayBuffer,
        save_cfg=True
    )

    # Please add your model path here.
    model_path = './DI-engine/dizoo/petting_zoo/config/ptz_simple_spread_mappo_seed0/ckpt/ckpt_best.pth.tar'

    evaluator_env = BaseEnvManager(
        env_fn=[lambda: env for _ in range(cfg.env.evaluator_env_num)],
        cfg=cfg.env.manager
    )
    evaluator_env.seed(seed, dynamic_seed=False)
    set_pkg_seed(seed, use_cuda=cfg.policy.cuda)

    model = MAVAC(**cfg.policy.model)
    policy = PPOPolicy(cfg.policy, model=model)
    policy.eval_mode.load_state_dict(torch.load(model_path, map_location='cpu'))
    tb_logger = SummaryWriter(os.path.join('./{}/log/'.format(cfg.exp_name), 'serial'))
    evaluator = InteractionSerialEvaluator(
        cfg.policy.eval.evaluator, evaluator_env, policy.eval_mode, tb_logger, exp_name=cfg.exp_name
    )
    evaluator.eval()


if __name__ == "__main__":
    main(main_config)
