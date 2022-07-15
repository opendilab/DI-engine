import torch
import torch.nn as nn
from ditk import logging
from ding.model import QAC, ConvEncoder
from ding.policy import SACPolicy
from ding.obs_model import CurlObsModel
from ding.envs import DingEnvWrapper, SubprocessEnvManagerV2
from ding.data import DequeBuffer
from ding.config import compile_config
from ding.framework import task
from ding.framework.context import OnlineRLContext
from ding.framework.middleware import data_pusher, StepCollector, interaction_evaluator, \
    CkptSaver, OffPolicyLearner, termination_checker, trainer
from ding.utils import set_pkg_seed
from dizoo.dmc2gym.envs.dmc2gym_env import DMC2GymEnv
from dizoo.dmc2gym.config.cartpole_swingup_curl_sac_config import main_config, create_config


def main():
    logging.getLogger().setLevel(logging.INFO)
    cfg = compile_config(main_config, create_cfg=create_config, auto=True)
    with task.start(async_mode=False, ctx=OnlineRLContext()):
        collector_env = SubprocessEnvManagerV2(
            env_fn=[lambda: DMC2GymEnv(cfg.env) for _ in range(cfg.env.collector_env_num)], cfg=cfg.env.manager
        )
        evaluator_env = SubprocessEnvManagerV2(
            env_fn=[lambda: DMC2GymEnv(cfg.env) for _ in range(cfg.env.evaluator_env_num)], cfg=cfg.env.manager
        )

        set_pkg_seed(cfg.seed, use_cuda=cfg.policy.cuda)

        obs_encoder = ConvEncoder(cfg.obs_model.obs_shape, hidden_size_list=cfg.obs_model.encoder_hidden_size_list)
        model = QAC(**cfg.policy.model, obs_encoder=obs_encoder)
        buffer_ = DequeBuffer(size=cfg.policy.other.replay_buffer.replay_buffer_size)
        policy = SACPolicy(cfg.policy, model=model)
        curl = CurlObsModel(cfg.obs_model, encoder=obs_encoder, encoder_target=policy._target_model.obs_encoder)

        task.use(interaction_evaluator(cfg, policy.eval_mode, evaluator_env))
        task.use(
            StepCollector(cfg, policy.collect_mode, collector_env, random_collect_size=cfg.policy.random_collect_size)
        )
        task.use(data_pusher(cfg, buffer_))
        task.use(trainer(cfg, curl))
        task.use(OffPolicyLearner(cfg, policy.learn_mode, buffer_))
        task.use(CkptSaver(cfg, policy, train_freq=1000))
        task.use(termination_checker(max_env_step=int(1e7)))
        task.run()


if __name__ == "__main__":
    main()
