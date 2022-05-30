"""
Main entry
"""
import gym
from ding.model import DQN
from ding.policy import DQNPolicy
from ding.envs import DingEnvWrapper, BaseEnvManager
from ding.data import DequeBuffer
from ding.config import compile_config
from ding.framework import task
from ding.framework.context import OnlineRLContext
from ding.framework.middleware import offpolicy_data_fetcher, trainer, step_collector, interaction_evaluator, \
    eps_greedy_handler, CkptSaver
from ding.utils import set_pkg_seed
from dizoo.classic_control.cartpole.config.cartpole_dqn_config import main_config, create_config


def main():
    cfg = compile_config(main_config, create_cfg=create_config, auto=True)
    with task.start(async_mode=False, ctx=OnlineRLContext()):
        collector_env = BaseEnvManager(
            env_fn=[lambda: DingEnvWrapper(gym.make("CartPole-v0")) for _ in range(8)], cfg=cfg.env.manager
        )
        evaluator_env = BaseEnvManager(
            env_fn=[lambda: DingEnvWrapper(gym.make("CartPole-v0")) for _ in range(5)], cfg=cfg.env.manager
        )

        set_pkg_seed(cfg.seed, use_cuda=cfg.policy.cuda)

        model = DQN(**cfg.policy.model)
        buffer_ = DequeBuffer(size=cfg.policy.other.replay_buffer.replay_buffer_size)
        policy = DQNPolicy(cfg.policy, model=model)

        task.use(interaction_evaluator(cfg, policy.eval_mode, evaluator_env))
        task.use(eps_greedy_handler(cfg))
        task.use(step_collector(cfg, policy.collect_mode, collector_env, buffer_))
        task.use(offpolicy_data_fetcher(cfg, buffer_))
        task.use(trainer(cfg, policy.learn_mode))
        task.use(CkptSaver(cfg, policy, train_freq=100))
        task.run(max_step=100000)


if __name__ == "__main__":
    main()
