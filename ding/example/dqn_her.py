import gym
from ditk import logging
from ding.model import DQN
from ding.policy import DQNPolicy
from ding.reward_model import HerRewardModel
from ding.envs import BaseEnvManagerV2
from ding.data import DequeBuffer
from ding.config import compile_config
from ding.framework import task
from ding.framework.context import OnlineRLContext
from ding.framework.middleware import HERLearner, EpisodeCollector, interaction_evaluator, data_pusher, \
    eps_greedy_handler, CkptSaver
from ding.utils import set_pkg_seed
from dizoo.bitflip.envs import BitFlipEnv
from dizoo.bitflip.config.bitflip_her_dqn_config import main_config, create_config


def main():
    logging.getLogger().setLevel(logging.INFO)
    cfg = compile_config(main_config, create_cfg=create_config, auto=True)
    with task.start(async_mode=False, ctx=OnlineRLContext()):
        collector_env = BaseEnvManagerV2(
            env_fn=[lambda: BitFlipEnv(cfg.env) for _ in range(cfg.env.collector_env_num)], cfg=cfg.env.manager
        )
        evaluator_env = BaseEnvManagerV2(
            env_fn=[lambda: BitFlipEnv(cfg.env) for _ in range(cfg.env.evaluator_env_num)], cfg=cfg.env.manager
        )

        set_pkg_seed(cfg.seed, use_cuda=cfg.policy.cuda)

        model = DQN(**cfg.policy.model)
        buffer_ = DequeBuffer(size=cfg.policy.other.replay_buffer.replay_buffer_size)
        policy = DQNPolicy(cfg.policy, model=model)
        her_reward_model = HerRewardModel(cfg.policy.other.her, cfg.policy.cuda)

        task.use(interaction_evaluator(cfg, policy.eval_mode, evaluator_env))
        task.use(eps_greedy_handler(cfg))
        task.use(EpisodeCollector(cfg, policy.collect_mode, collector_env))
        task.use(data_pusher(cfg, buffer_))
        task.use(HERLearner(cfg, policy.learn_mode, buffer_, her_reward_model))
        task.use(CkptSaver(cfg, policy, train_freq=100))
        task.run()


if __name__ == "__main__":
    main()
