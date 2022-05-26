import gym
from ditk import logging
import torch
from ding.model import DQN
from ding.policy import SQLPolicy
from ding.envs import DingEnvWrapper, BaseEnvManagerV2
from ding.data import DequeBuffer
from ding.config import compile_config
from ding.framework import task
from ding.framework.context import OnlineRLContext
from ding.framework.middleware import OffPolicyLearner, StepCollector, interaction_evaluator, \
    eps_greedy_handler, CkptSaver, eps_greedy_masker, sqil_data_pusher
from ding.utils import set_pkg_seed
from dizoo.classic_control.cartpole.config.cartpole_sql_config import main_config as ex_main_config
from dizoo.classic_control.cartpole.config.cartpole_sql_config import create_config as ex_create_config
from dizoo.classic_control.cartpole.config.cartpole_sqil_config import main_config, create_config


def main():
    logging.getLogger().setLevel(logging.INFO)
    cfg = compile_config(main_config, create_cfg=create_config, auto=True)
    expert_cfg = compile_config(ex_main_config, create_cfg=ex_create_config, auto=True)
    # expert config must have the same `n_sample`. The line below ensure we do not need to modify the expert configs
    expert_cfg.policy.collect.n_sample = cfg.policy.collect.n_sample
    with task.start(async_mode=False, ctx=OnlineRLContext()):
        collector_env = BaseEnvManagerV2(
            env_fn=[lambda: DingEnvWrapper(gym.make("CartPole-v0")) for _ in range(cfg.env.collector_env_num)],
            cfg=cfg.env.manager
        )
        expert_collector_env = BaseEnvManagerV2(
            env_fn=[lambda: DingEnvWrapper(gym.make("CartPole-v0")) for _ in range(cfg.env.collector_env_num)],
            cfg=cfg.env.manager
        )
        evaluator_env = BaseEnvManagerV2(
            env_fn=[lambda: DingEnvWrapper(gym.make("CartPole-v0")) for _ in range(cfg.env.evaluator_env_num)],
            cfg=cfg.env.manager
        )

        set_pkg_seed(cfg.seed, use_cuda=cfg.policy.cuda)

        model = DQN(**cfg.policy.model)
        expert_model = DQN(**cfg.policy.model)

        buffer_ = DequeBuffer(size=cfg.policy.other.replay_buffer.replay_buffer_size)
        expert_buffer = DequeBuffer(size=cfg.policy.other.replay_buffer.replay_buffer_size)

        policy = SQLPolicy(cfg.policy, model=model)
        expert_policy = SQLPolicy(expert_cfg.policy, model=expert_model)
        state_dict = torch.load(cfg.policy.collect.model_path, map_location='cpu')
        expert_policy.collect_mode.load_state_dict(state_dict)

        task.use(interaction_evaluator(cfg, policy.eval_mode, evaluator_env))
        task.use(eps_greedy_handler(cfg))
        task.use(StepCollector(cfg, policy.collect_mode, collector_env))  # agent data collector
        task.use(sqil_data_pusher(cfg, buffer_, expert=False))
        task.use(eps_greedy_masker())
        task.use(StepCollector(cfg, expert_policy.collect_mode, expert_collector_env))  # expert data collector
        task.use(sqil_data_pusher(cfg, expert_buffer, expert=True))
        task.use(OffPolicyLearner(cfg, policy.learn_mode, [(buffer_, 0.5), (expert_buffer, 0.5)]))
        task.use(CkptSaver(cfg, policy, train_freq=100))
        task.run()


if __name__ == "__main__":
    main()
