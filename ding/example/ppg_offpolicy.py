import gym
from ditk import logging
from ding.model import PPG
from ding.policy import PPGPolicy
from ding.envs import DingEnvWrapper, BaseEnvManagerV2
from ding.data import DequeBuffer
from ding.data.buffer.middleware import use_time_check, sample_range_view
from ding.config import compile_config
from ding.framework import task
from ding.framework.context import OnlineRLContext
from ding.framework.middleware import OffPolicyLearner, StepCollector, interaction_evaluator, data_pusher, \
    CkptSaver, gae_estimator
from ding.utils import set_pkg_seed
from dizoo.classic_control.cartpole.config.cartpole_ppg_config import main_config, create_config


def main():
    logging.getLogger().setLevel(logging.INFO)
    cfg = compile_config(main_config, create_cfg=create_config, auto=True)
    with task.start(async_mode=False, ctx=OnlineRLContext()):
        collector_env = BaseEnvManagerV2(
            env_fn=[lambda: DingEnvWrapper(gym.make("CartPole-v0")) for _ in range(cfg.env.collector_env_num)],
            cfg=cfg.env.manager
        )
        evaluator_env = BaseEnvManagerV2(
            env_fn=[lambda: DingEnvWrapper(gym.make("CartPole-v0")) for _ in range(cfg.env.evaluator_env_num)],
            cfg=cfg.env.manager
        )

        set_pkg_seed(cfg.seed, use_cuda=cfg.policy.cuda)

        model = PPG(**cfg.policy.model)
        buffer_cfg = cfg.policy.other.replay_buffer
        max_size = max(buffer_cfg.policy.replay_buffer_size, buffer_cfg.value.replay_buffer_size)
        buffer_ = DequeBuffer(size=max_size)
        policy_buffer = buffer_.view()  # shallow copy
        policy_buffer.use(use_time_check(policy_buffer, max_use=buffer_cfg.policy.max_use))
        policy_buffer.use(sample_range_view(policy_buffer, start=-buffer_cfg.policy.replay_buffer_size))
        value_buffer = buffer_.view()
        value_buffer.use(use_time_check(value_buffer, max_use=buffer_cfg.value.max_use))
        value_buffer.use(sample_range_view(value_buffer, start=-buffer_cfg.value.replay_buffer_size))
        policy = PPGPolicy(cfg.policy, model=model)

        task.use(interaction_evaluator(cfg, policy.eval_mode, evaluator_env))
        task.use(StepCollector(cfg, policy.collect_mode, collector_env))
        task.use(gae_estimator(cfg, policy.collect_mode, buffer_))
        task.use(OffPolicyLearner(cfg, policy.learn_mode, {'policy': policy_buffer, 'value': value_buffer}))
        task.use(CkptSaver(cfg, policy, train_freq=100))
        task.run()


if __name__ == "__main__":
    main()
