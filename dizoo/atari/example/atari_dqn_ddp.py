from copy import deepcopy
from ditk import logging
from ding.model import DQN
from ding.policy import DQNPolicy
from ding.envs import DingEnvWrapper, SubprocessEnvManagerV2
from ding.data import DequeBuffer
from ding.config import compile_config
from ding.utils import DistContext, get_rank
from ding.framework import task
from ding.framework.context import OnlineRLContext
from ding.framework.middleware import OffPolicyLearner, StepCollector, interaction_evaluator, data_pusher, \
    eps_greedy_handler, CkptSaver, nstep_reward_enhancer
from ding.utils import set_pkg_seed
from dizoo.atari.envs.atari_env import AtariEnv
from dizoo.atari.config.serial.pong.pong_dqn_config import main_config, create_config


def ddp_termination_checker(max_env_step=None, max_train_iter=None, rank=0):
    import numpy as np
    import torch
    from ding.utils import broadcast
    if rank == 0:
        if max_env_step is None:
            max_env_step = np.inf
        if max_train_iter is None:
            max_train_iter = np.inf

    def _check(ctx):
        if rank == 0:
            if ctx.env_step > max_env_step:
                finish = torch.ones(1).long().cuda()
            elif ctx.train_iter > max_train_iter:
                finish = torch.ones(1).long().cuda()
            else:
                finish = torch.zeros(1).long().cuda()
        else:
            finish = torch.zeros(1).long().cuda()
        broadcast(finish, 0)
        task.finish = finish.cpu().bool().item()

    return _check


def main():
    logging.getLogger().setLevel(logging.INFO)
    main_config.exp_name = 'pong_dqn_seed0_ddp'
    main_config.policy.learn.multi_gpu = True
    cfg = compile_config(main_config, create_cfg=create_config, auto=True)
    with DistContext():
        rank = get_rank()
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

            set_pkg_seed(cfg.seed, use_cuda=cfg.policy.cuda)

            model = DQN(**cfg.policy.model)
            buffer_ = DequeBuffer(size=cfg.policy.other.replay_buffer.replay_buffer_size)
            policy = DQNPolicy(cfg.policy, model=model)

            if rank == 0:
                task.use(interaction_evaluator(cfg, policy.eval_mode, evaluator_env))
            task.use(eps_greedy_handler(cfg))
            task.use(StepCollector(cfg, policy.collect_mode, collector_env))
            task.use(nstep_reward_enhancer(cfg))
            task.use(data_pusher(cfg, buffer_))
            task.use(OffPolicyLearner(cfg, policy.learn_mode, buffer_))
            if rank == 0:
                task.use(CkptSaver(cfg, policy, train_freq=1000))
            task.use(ddp_termination_checker(max_train_iter=int(1e7), rank=rank))
            task.run()


if __name__ == "__main__":
    main()
