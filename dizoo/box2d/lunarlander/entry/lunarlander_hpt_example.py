import gym
import torch
import torch.nn as nn
from ditk import logging
from ding.data.model_loader import FileModelLoader
from ding.data.storage_loader import FileStorageLoader
from ding.model.common.head import DuelingHead
from ding.model.template.hpt import HPT
from ding.policy import DQNPolicy
from ding.envs import DingEnvWrapper, SubprocessEnvManagerV2
from ding.data import DequeBuffer
from ding.config import compile_config
from ding.framework import task, ding_init
from ding.framework.context import OnlineRLContext
from ding.framework.middleware import OffPolicyLearner, StepCollector, interaction_evaluator, data_pusher, \
    eps_greedy_handler, CkptSaver, ContextExchanger, ModelExchanger, online_logger, termination_checker, \
    nstep_reward_enhancer
from ding.utils import set_pkg_seed
from dizoo.box2d.lunarlander.config.lunarlander_hpt_config import main_config, create_config


def main():
    logging.getLogger().setLevel(logging.INFO)
    cfg = compile_config(main_config, create_cfg=create_config, auto=True, save_cfg=task.router.node_id == 0)
    ding_init(cfg)

    with task.start(async_mode=False, ctx=OnlineRLContext()):
        collector_env = SubprocessEnvManagerV2(
            env_fn=[lambda: DingEnvWrapper(gym.make("LunarLander-v2")) for _ in range(cfg.env.collector_env_num)],
            cfg=cfg.env.manager
        )
        evaluator_env = SubprocessEnvManagerV2(
            env_fn=[lambda: DingEnvWrapper(gym.make("LunarLander-v2")) for _ in range(cfg.env.evaluator_env_num)],
            cfg=cfg.env.manager
        )

        set_pkg_seed(cfg.seed, use_cuda=cfg.policy.cuda)

        # Migrating models to the GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # HPT introduces a Policy Stem module, which processes the input features using Cross-Attention.
        model = HPT(cfg.policy.model.obs_shape, cfg.policy.model.action_shape).to(device)
        buffer_ = DequeBuffer(size=cfg.policy.other.replay_buffer.replay_buffer_size)

        # Pass the model into Policy
        policy = DQNPolicy(cfg.policy, model=model)

        # Consider the case with multiple processes
        if task.router.is_active:
            # You can use labels to distinguish between workers with different roles,
            # here we use node_id to distinguish.
            if task.router.node_id == 0:
                task.add_role(task.role.LEARNER)
            elif task.router.node_id == 1:
                task.add_role(task.role.EVALUATOR)
            else:
                task.add_role(task.role.COLLECTOR)

            # Sync their context and model between each worker.
            task.use(ContextExchanger(skip_n_iter=1))
            task.use(ModelExchanger(model))

        # Here is the part of single process pipeline.
        task.use(interaction_evaluator(cfg, policy.eval_mode, evaluator_env))
        task.use(eps_greedy_handler(cfg))
        task.use(StepCollector(cfg, policy.collect_mode, collector_env))
        task.use(nstep_reward_enhancer(cfg))
        task.use(data_pusher(cfg, buffer_))
        task.use(OffPolicyLearner(cfg, policy.learn_mode, buffer_))
        task.use(online_logger(train_show_freq=50))
        task.use(CkptSaver(policy, cfg.exp_name, train_freq=1000))
        task.use(termination_checker(max_env_step=int(3e6)))
        task.run()


if __name__ == "__main__":
    main()
