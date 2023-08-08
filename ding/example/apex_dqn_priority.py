import gym
from ditk import logging
from ding.model import DQN
from ding.policy import DQNPolicy
from ding.envs import DingEnvWrapper, BaseEnvManagerV2
from ding.data import DequeBuffer
from ding.data.buffer.middleware import PriorityExperienceReplay
from ding.config import compile_config
from ding.framework import task
from ding.framework.context import OnlineRLContext
from ding.framework.middleware import OffPolicyLearner, StepCollector, interaction_evaluator, data_pusher, \
    eps_greedy_handler, CkptSaver, nstep_reward_enhancer, priority_calculator
from ding.utils import set_pkg_seed
from dizoo.classic_control.cartpole.config.cartpole_dqn_config import main_config, create_config


def main():
    logging.getLogger().setLevel(logging.INFO)
    main_config.exp_name = 'cartpole_dqn_per'
    main_config.policy.priority = True
    main_config.policy.priority_IS_weight = True
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
        buffer_.use(PriorityExperienceReplay(buffer_, IS_weight=True))
        policy = DQNPolicy(cfg.policy, model=model)

        task.use(interaction_evaluator(cfg, policy.eval_mode, evaluator_env))
        task.use(eps_greedy_handler(cfg))
        task.use(StepCollector(cfg, policy.collect_mode, collector_env))
        if "nstep" in cfg.policy and cfg.policy.nstep > 1:
            task.use(nstep_reward_enhancer(cfg))

        def dqn_priority_calculation(update_target_model_frequency):
            last_update_train_iter = 0

            def _calculate_priority(data):
                nonlocal last_update_train_iter

                if (task.ctx.train_iter - last_update_train_iter) % update_target_model_frequency == 0:
                    update_target_model = True
                else:
                    update_target_model = False
                priority = policy.calculate_priority(data, update_target_model=update_target_model)['priority']
                last_update_train_iter = task.ctx.train_iter
                return priority

            return _calculate_priority

        task.use(
            priority_calculator(
                priority_calculation_fn=dqn_priority_calculation(
                    update_target_model_frequency=cfg.policy.learn.target_update_freq
                ),
            )
        )
        task.use(data_pusher(cfg, buffer_))
        task.use(OffPolicyLearner(cfg, policy.learn_mode, buffer_))
        task.use(CkptSaver(policy, cfg.exp_name, train_freq=100))
        task.run()


if __name__ == "__main__":
    main()
