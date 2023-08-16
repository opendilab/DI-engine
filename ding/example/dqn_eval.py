import gym
import torch
from ditk import logging
from ding.model import DQN
from ding.policy import DQNPolicy
from ding.envs import DingEnvWrapper, BaseEnvManagerV2
from ding.config import compile_config
from ding.framework import task
from ding.framework.context import OnlineRLContext
from ding.framework.middleware import interaction_evaluator
from ding.utils import set_pkg_seed
from dizoo.classic_control.cartpole.config.cartpole_dqn_config import main_config, create_config


def main():
    logging.getLogger().setLevel(logging.INFO)
    main_config.exp_name = 'cartpole_dqn_eval'
    cfg = compile_config(main_config, create_cfg=create_config, auto=True)
    with task.start(async_mode=False, ctx=OnlineRLContext()):
        evaluator_env = BaseEnvManagerV2(
            env_fn=[lambda: DingEnvWrapper(gym.make("CartPole-v0")) for _ in range(cfg.env.evaluator_env_num)],
            cfg=cfg.env.manager
        )

        set_pkg_seed(cfg.seed, use_cuda=cfg.policy.cuda)
        model = DQN(**cfg.policy.model)

        # Load the pretrained weights.
        # First, you should get a pretrained network weights.
        # For example, you can run ``python3 -u ding/examples/dqn.py``.
        pretrained_state_dict = torch.load('cartpole_dqn_seed0/ckpt/final.pth.tar', map_location='cpu')['model']
        model.load_state_dict(pretrained_state_dict)

        policy = DQNPolicy(cfg.policy, model=model)

        # Define the evaluator middleware.
        task.use(interaction_evaluator(cfg, policy.eval_mode, evaluator_env))
        task.run(max_step=1)


if __name__ == "__main__":
    main()
