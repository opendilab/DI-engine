import gym
from tensorboardX import SummaryWriter
import copy
import easydict
import os
from ditk import logging

from ding.model import DQN
from ding.policy import DQNPolicy
from ding.envs import DingEnvWrapper, BaseEnvManagerV2
from ding.data import DequeBuffer
from ding.config import compile_config
from ding.framework import task
from ding.framework.context import OnlineRLContext
from ding.framework.middleware import OffPolicyLearner, StepCollector, interaction_evaluator, \
    eps_greedy_handler, CkptSaver, eps_greedy_masker, sqil_data_pusher, data_pusher
from ding.utils import set_pkg_seed
from dizoo.classic_control.cartpole.config.cartpole_trex_dqn_config import main_config, create_config
from ding.entry import trex_collecting_data
from ding.reward_model import create_reward_model


def main():
    logging.getLogger().setLevel(logging.INFO)
    demo_arg = easydict.EasyDict({'cfg': [main_config, create_config], 'seed': 0})
    trex_collecting_data(demo_arg)
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
        model = DQN(**cfg.policy.model)
        buffer_ = DequeBuffer(size=cfg.policy.other.replay_buffer.replay_buffer_size)
        policy = DQNPolicy(cfg.policy, model=model)

        tb_logger = SummaryWriter(os.path.join('./{}/log/'.format(cfg.exp_name), 'serial'))
        reward_model = create_reward_model(copy.deepcopy(cfg), policy.collect_mode.get_attribute('device'), tb_logger)
        reward_model.train()

        task.use(interaction_evaluator(cfg, policy.eval_mode, evaluator_env))
        task.use(eps_greedy_handler(cfg))
        task.use(StepCollector(cfg, policy.collect_mode, collector_env))
        task.use(data_pusher(cfg, buffer_))
        task.use(OffPolicyLearner(cfg, policy.learn_mode, buffer_, reward_model))
        task.use(CkptSaver(cfg, policy, train_freq=100))
        task.run()


if __name__ == "__main__":
    main()
