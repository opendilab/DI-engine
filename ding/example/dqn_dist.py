"""
The distributed version of DQN pipeline.
With N workers = 1 learner + 1 evaluator + (N-2) actors

# First Example —— Execute on one machine with multi processes.
Execute 4 processes with 1 learner + 1 evaluator + 2 actors
Remember to keep them connected by mesh to ensure that they can exchange information with each other.

> ditask --package . --main ding.example.dqn_dist.main --parallel-workers 4 --topology mesh

# Second Example —— Execute on multiple machines.
1. Execute 1 learner + 1 evaluator on one machine.

> ditask --package . --main ding.example.dqn_dist.main --parallel-workers 2 --topology mesh --node-ids 0 --ports 50515

2. Execute 2 collectors on another machine. (Suppose the ip of the first machine is 127.0.0.1).
    Here we use `alone` topology instead of `mesh` because the collectors do not need communicate with each other.
    Remember the `node_ids` cannot be duplicated with the learner, evaluator processes.
    And remember to set the `ports` (should not conflict with others) and `attach_to` parameters.
    The value of the `attach_to` parameter should be obtained from the log of the
    process started earlier (e.g. 'NNG listen on tcp://10.0.0.4:50515').

> ditask --package . --main ding.example.dqn_dist.main --parallel-workers 2 --topology alone --node-ids 2 \
    --ports 50517 --attach-to tcp://10.0.0.4:50515,tcp://127.0.0.1:50516

3. You can repeat step 2 to start more collectors on other machines.
"""
import gym
import logging
from ding.model import DQN
from ding.policy import DQNPolicy
from ding.envs import DingEnvWrapper, BaseEnvManagerV2
from ding.data import DequeBuffer
from ding.config import compile_config
from ding.framework import task
from ding.framework.context import OnlineRLContext
from ding.framework.middleware import OffPolicyLearner, StepCollector, interaction_evaluator, data_pusher, \
    eps_greedy_handler, CkptSaver, context_exchanger, model_exchanger
from ding.utils import set_pkg_seed
from dizoo.classic_control.cartpole.config.cartpole_dqn_config import main_config, create_config


def main():
    logging.getLogger().setLevel(logging.INFO)
    cfg = compile_config(main_config, create_cfg=create_config, auto=True)
    # cfg.env.stop_value = 99999999  # Don't stop
    with task.start(async_mode=False, ctx=OnlineRLContext()):
        assert task.router.is_active, "Please execute this script with ditask! See note in the header."

        set_pkg_seed(cfg.seed, use_cuda=cfg.policy.cuda)

        model = DQN(**cfg.policy.model)
        buffer_ = DequeBuffer(size=cfg.policy.other.replay_buffer.replay_buffer_size)
        policy = DQNPolicy(cfg.policy, model=model)

        if task.router.node_id == 0:  # Learner
            logging.info("Learner running on node {}".format(task.router.node_id))
            buffer_ = DequeBuffer(size=cfg.policy.other.replay_buffer.replay_buffer_size)
            task.use(
                context_exchanger(
                    send_keys=["train_iter"],
                    recv_keys=["trajectories", "episodes", "env_step", "env_episode"],
                    skip_n_iter=0
                )
            )
            task.use(model_exchanger(model, is_learner=True))
            task.use(data_pusher(cfg, buffer_))
            task.use(OffPolicyLearner(cfg, policy.learn_mode, buffer_))
            task.use(CkptSaver(cfg, policy, train_freq=100))

        elif task.router.node_id == 1:  # Evaluator
            logging.info("Evaluator running on node {}".format(task.router.node_id))
            evaluator_env = BaseEnvManagerV2(
                env_fn=[lambda: DingEnvWrapper(gym.make("CartPole-v0")) for _ in range(cfg.env.evaluator_env_num)],
                cfg=cfg.env.manager
            )
            task.use(context_exchanger(recv_keys=["train_iter"], skip_n_iter=1))
            task.use(model_exchanger(model, is_learner=False))
            task.use(interaction_evaluator(cfg, policy.eval_mode, evaluator_env))
            task.use(CkptSaver(cfg, policy, save_finish=False))

        else:  # Collectors
            logging.info("Collector running on node {}".format(task.router.node_id))
            collector_env = BaseEnvManagerV2(
                env_fn=[lambda: DingEnvWrapper(gym.make("CartPole-v0")) for _ in range(cfg.env.collector_env_num)],
                cfg=cfg.env.manager
            )
            task.use(
                context_exchanger(
                    send_keys=["trajectories", "episodes", "env_step", "env_episode"],
                    recv_keys=["train_iter"],
                    skip_n_iter=1
                )
            )
            task.use(model_exchanger(model, is_learner=False))
            task.use(eps_greedy_handler(cfg))
            task.use(StepCollector(cfg, policy.collect_mode, collector_env))

        task.run()


if __name__ == "__main__":
    main()
