import sys
import numpy as np
from copy import deepcopy
from ditk import logging
from ding.model import DQN
from ding.policy import DQNPolicy
from ding.envs import SubprocessEnvManagerV2
from ding.data import DequeBuffer
from ding.config import compile_config
from ding.framework import task
from ding.framework.context import OnlineRLContext
from ding.framework.middleware import OffPolicyLearner, StepCollector, interaction_evaluator, data_pusher, \
    eps_greedy_handler, nstep_reward_enhancer, termination_checker
from ding.utils import set_pkg_seed
from ding.utils.log_helper import build_logger
from dizoo.atari.envs.atari_env import AtariEnv
from pong_dqn_errorclip_config import main_config, create_config

# seed = [0]
# num_seed=len(seed)

def main(seed):
    num_seed = 1
    logging.getLogger().setLevel(logging.INFO)
    cfg = compile_config(main_config, create_cfg=create_config, auto=True)

    for i in range(num_seed):
        logger, tb_logger = build_logger(path='../log/pong_dqn_errorclip/seed'+str(seed), need_tb=True)
        with task.start(async_mode=False, ctx=OnlineRLContext()):
            collector_cfg = deepcopy(cfg.env)
            collector_cfg.is_train = True
            evaluator_cfg = deepcopy(cfg.env)
            evaluator_cfg.is_train = False
            collector_env = SubprocessEnvManagerV2(
                env_fn=[lambda: AtariEnv(collector_cfg) for _ in range(cfg.env.collector_env_num)], cfg=cfg.env.manager
            )
            evaluator_env = SubprocessEnvManagerV2(
                env_fn=[lambda: AtariEnv(evaluator_cfg) for _ in range(cfg.env.evaluator_env_num)], cfg=cfg.env.manager
            )

            set_pkg_seed(seed, use_cuda=cfg.policy.cuda)

            model = DQN(**cfg.policy.model)
            buffer_ = DequeBuffer(size=cfg.policy.other.replay_buffer.replay_buffer_size)
            policy = DQNPolicy(cfg.policy, model=model)

            def _add_scalar(ctx):
                if ctx.eval_value != -np.inf:
                    tb_logger.add_scalar('evaluator_step/reward',ctx.eval_value, global_step= ctx.env_step)

            task.use(interaction_evaluator(cfg, policy.eval_mode, evaluator_env))
            task.use(eps_greedy_handler(cfg))
            task.use(StepCollector(cfg, policy.collect_mode, collector_env))
            task.use(nstep_reward_enhancer(cfg))
            task.use(data_pusher(cfg, buffer_))
            task.use(OffPolicyLearner(cfg, policy.learn_mode, buffer_))
            task.use(termination_checker(max_env_step=int(10e6)))
            task.use(_add_scalar)
            task.run()

if __name__ == "__main__":
    main(int(sys.argv[1]))
