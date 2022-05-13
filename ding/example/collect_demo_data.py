import gym
from ditk import logging
import torch
from ding.model import QAC
from ding.policy import SACPolicy
from ding.envs import DingEnvWrapper, BaseEnvManagerV2
from ding.data import offline_data_save_type
from ding.config import compile_config
from ding.framework import task
from ding.framework.context import OnlineRLContext
from ding.framework.middleware import StepCollector, offline_data_saver
from ding.utils import set_pkg_seed
from dizoo.classic_control.pendulum.envs.pendulum_env import PendulumEnv
from dizoo.classic_control.pendulum.config.pendulum_sac_data_generation_config import main_config, create_config


def main():
    logging.getLogger().setLevel(logging.INFO)
    cfg = compile_config(main_config, create_cfg=create_config, auto=True, evaluator=None)
    with task.start(async_mode=False, ctx=OnlineRLContext()):
        collector_env = BaseEnvManagerV2(env_fn=[lambda: PendulumEnv(cfg.env) for _ in range(10)], cfg=cfg.env.manager)

        set_pkg_seed(cfg.seed, use_cuda=cfg.policy.cuda)

        model = QAC(**cfg.policy.model)
        policy = SACPolicy(cfg.policy, model=model, enable_field=['collect'])
        state_dict = torch.load(cfg.policy.collect.state_dict_path, map_location='cpu')
        policy.collect_mode.load_state_dict(state_dict)

        task.use(StepCollector(cfg, policy.collect_mode, collector_env))
        task.use(offline_data_saver(cfg, cfg.policy.collect.save_path, data_type='hdf5'))
        task.run(max_step=1)


if __name__ == "__main__":
    main()
