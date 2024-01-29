import torch.nn as nn
import torch.distributed as dist
from ditk import logging
from ding.model import DecisionTransformer
from ding.policy import DTPolicy
from ding.envs import SubprocessEnvManagerV2
from ding.envs import AllinObsWrapper
from ding.data import create_dataset
from ding.config import compile_config
from ding.framework import task, ding_init
from ding.framework.context import OfflineRLContext
from ding.framework.middleware import interaction_evaluator, trainer, CkptSaver, offline_logger, termination_checker, \
    OfflineMemoryDataFetcher
from ding.utils import set_pkg_seed, DDPContext, to_ddp_config
from dizoo.atari.envs import AtariEnv
from dizoo.atari.config.serial.pong.pong_dt_config import main_config, create_config


def main():
    # If you don't have offline data, you need to prepare if first and set the data_path in config
    # For demostration, we also can train a RL policy (e.g. SAC) and collect some data
    logging.getLogger().setLevel(logging.INFO)
    with DDPContext():
        cmain_config = to_ddp_config(main_config)
        cfg = compile_config(cmain_config, create_cfg=create_config, auto=True)
        ding_init(cfg)
        with task.start(async_mode=False, ctx=OfflineRLContext()):
            evaluator_env = SubprocessEnvManagerV2(
                env_fn=[lambda: AllinObsWrapper(AtariEnv(cfg.env)) for _ in range(cfg.env.evaluator_env_num)],
                cfg=cfg.env.manager
            )

            set_pkg_seed(cfg.seed, use_cuda=cfg.policy.cuda)
            dataset = create_dataset(cfg)
            cfg.policy.model.max_timestep = dataset.get_max_timestep()
            state_encoder = nn.Sequential(
                nn.Conv2d(4, 32, 8, stride=4, padding=0), nn.ReLU(), nn.Conv2d(32, 64, 4, stride=2, padding=0),
                nn.ReLU(), nn.Conv2d(64, 64, 3, stride=1, padding=0), nn.ReLU(), nn.Flatten(),
                nn.Linear(3136, cfg.policy.model.h_dim), nn.Tanh()
            )

            model = DecisionTransformer(**cfg.policy.model, state_encoder=state_encoder)
            # model.parallelize()
            policy = DTPolicy(cfg.policy, model=model)

            task.use(interaction_evaluator(cfg, policy.eval_mode, evaluator_env))
            task.use(OfflineMemoryDataFetcher(cfg, dataset))
            task.use(trainer(cfg, policy.learn_mode))
            task.use(termination_checker(max_train_iter=3e4))
            task.use(CkptSaver(policy, cfg.exp_name, train_freq=100))
            task.use(offline_logger())
            task.run()


if __name__ == "__main__":
    main()
