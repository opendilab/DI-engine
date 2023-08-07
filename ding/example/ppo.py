from capture_class import my_hook
import gym
# from capture_eiengine import insert_capture

# import torch_dipu
from ditk import logging
from ding.model import VAC
from ding.policy import PPOPolicy
from ding.envs import DingEnvWrapper, BaseEnvManagerV2
from ding.data import DequeBuffer
from ding.config import compile_config
from ding.framework import task, ding_init
from ding.framework.context import OnlineRLContext
from ding.framework.middleware import multistep_trainer, StepCollector, interaction_evaluator, CkptSaver, \
    gae_estimator, online_logger
from ding.utils import set_pkg_seed
from dizoo.classic_control.cartpole.config.cartpole_ppo_config import main_config, create_config
import os


def main():
    logging.getLogger().setLevel(logging.INFO)
    cfg = compile_config(main_config, create_cfg=create_config, auto=True)
    if( os.getenv('ONE_ITER_TOOL_DEVICE', None) != "cpu"):
        cfg['policy']['cuda']=True
    else:
        cfg['policy']['cuda']=False
    # cfg['seed']=5
    # cfg['policy']['cuda']=True
    # cfg['env']['collector_env_num']=1
    
    # cfg['policy']['cuda']=True
    ding_init(cfg)
    cfg.seed = 5
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

        model = VAC(**cfg.policy.model)
        policy = PPOPolicy(cfg.policy, model=model)
        my_hook.insert_capture(policy)

        task.use(interaction_evaluator(cfg, policy.eval_mode, evaluator_env))
        task.use(StepCollector(cfg, policy.collect_mode, collector_env))
        task.use(gae_estimator(cfg, policy.collect_mode))
        task.use(multistep_trainer(policy.learn_mode, log_freq=50))
        task.use(CkptSaver(policy, cfg.exp_name, train_freq=100))
        task.use(online_logger(train_show_freq=3))
        task.run()


if __name__ == "__main__":
    main()
