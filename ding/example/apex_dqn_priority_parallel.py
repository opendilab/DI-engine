import os
from ditk import logging
from ding.model import DQN
from ding.policy import DQNPolicy
from ding.data import DequeBuffer
from ding.data.buffer.middleware import PriorityExperienceReplay
from ding.envs import setup_ding_env_manager
from ding.config import compile_config
from ding.framework import task, ding_init
from ding.framework import Parallel
from ding.framework.context import OnlineRLContext
from ding.framework.middleware import OffPolicyLearner, StepCollector, interaction_evaluator, data_pusher, \
    eps_greedy_handler, CkptSaver, ModelExchanger, ContextExchanger, online_logger, \
    nstep_reward_enhancer, priority_calculator
from ding.utils import set_pkg_seed


def main():
    from ding.config.DQN.gym_lunarlander_v2 import cfg, env

    cfg.exp_name = 'LunarLander-v2-Apex-DQN'
    cfg.policy.priority = True
    cfg.policy.priority_IS_weight = True
    cfg = compile_config(cfg, policy=DQNPolicy, save_cfg=task.router.node_id == 0)

    logging.getLogger().setLevel(logging.INFO)
    model_path = os.path.join(cfg.exp_name, 'models')
    ding_init(cfg)
    with task.start(async_mode=False, ctx=OnlineRLContext()):

        assert task.router.is_active
        if task.router.node_id == 0:
            task.add_role(task.role.LEARNER)
        elif task.router.node_id == 1:
            task.add_role(task.role.EVALUATOR)
        else:
            task.add_role(task.role.COLLECTOR)

        set_pkg_seed(cfg.seed, use_cuda=cfg.policy.cuda)

        if task.has_role(task.role.COLLECTOR):
            collector_env = setup_ding_env_manager(env(cfg=cfg.env), cfg.env.collector_env_num, 'collector', debug=True)
            evaluator_env = setup_ding_env_manager(env(cfg=cfg.env), cfg.env.evaluator_env_num, 'evaluator', debug=True)
        elif task.has_role(task.role.EVALUATOR):
            collector_env = setup_ding_env_manager(env(cfg=cfg.env), cfg.env.collector_env_num, 'collector', debug=True)
            evaluator_env = setup_ding_env_manager(env(cfg=cfg.env), cfg.env.evaluator_env_num, 'evaluator', debug=True)
        elif task.has_role(task.role.LEARNER):
            collector_env = setup_ding_env_manager(env(cfg=cfg.env), cfg.env.collector_env_num, 'collector', debug=True)
            evaluator_env = setup_ding_env_manager(env(cfg=cfg.env), cfg.env.evaluator_env_num, 'evaluator', debug=True)

        if task.has_role(task.role.LEARNER):
            model = DQN(**cfg.policy.model)
            buffer_ = DequeBuffer(size=cfg.policy.other.replay_buffer.replay_buffer_size)
            buffer_.use(PriorityExperienceReplay(buffer_, IS_weight=True))
            policy = DQNPolicy(cfg.policy, model=model)
            task.use(ContextExchanger(skip_n_iter=1))
            task.use(ModelExchanger(model))

        elif task.has_role(task.role.COLLECTOR):
            model = DQN(**cfg.policy.model)
            buffer_ = DequeBuffer(size=cfg.policy.other.replay_buffer.replay_buffer_size)
            buffer_.use(PriorityExperienceReplay(buffer_, IS_weight=True))
            policy = DQNPolicy(cfg.policy, model=model)
            task.use(ContextExchanger(skip_n_iter=1))
            task.use(ModelExchanger(model))

        elif task.has_role(task.role.EVALUATOR):
            model = DQN(**cfg.policy.model)
            buffer_ = DequeBuffer(size=cfg.policy.other.replay_buffer.replay_buffer_size)
            buffer_.use(PriorityExperienceReplay(buffer_, IS_weight=True))
            policy = DQNPolicy(cfg.policy, model=model)
            task.use(ContextExchanger(skip_n_iter=1))
            task.use(ModelExchanger(model))

        # Here is the part of single process pipeline.
        task.use(interaction_evaluator(cfg, policy.eval_mode, evaluator_env))
        task.use(eps_greedy_handler(cfg))
        task.use(StepCollector(cfg, policy.collect_mode, collector_env))
        print(f"cfg.policy.nstep:{cfg.policy.nstep}")
        if "nstep" in cfg.policy and cfg.policy.nstep > 1:
            if task.has_role(task.role.COLLECTOR):
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
                func_for_priority_calculation=dqn_priority_calculation(
                    update_target_model_frequency=cfg.policy.learn.target_update_freq
                ),
            )
        )

        task.use(data_pusher(cfg, buffer_))
        task.use(OffPolicyLearner(cfg, policy.learn_mode, buffer_))
        task.use(online_logger(train_show_freq=10))
        task.use(CkptSaver(policy, cfg.exp_name, train_freq=100))
        task.run()


if __name__ == "__main__":

    Parallel.runner(
        n_parallel_workers=3,
        ports=50515,
        protocol="tcp",
        topology="mesh",
        attach_to=None,
        address=None,
        labels=None,
        node_ids=None,
        mq_type="nng",
        redis_host=None,
        redis_port=None,
        startup_interval=1
    )(main)
