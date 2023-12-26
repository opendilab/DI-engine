import gym
import torch
from easydict import EasyDict

from ding.config import compile_config
from ding.envs import DingEnvWrapper
from ding.model import DQN
from ding.policy import DQNPolicy, single_env_forward_wrapper
from dizoo.cliffwalking.config.cliffwalking_dqn_config import create_config, main_config
from dizoo.cliffwalking.envs.cliffwalking_env import CliffWalkingEnv


def main(main_config: EasyDict, create_config: EasyDict, ckpt_path: str):
    main_config.exp_name = f'cliffwalking_dqn_seed0_deploy'
    cfg = compile_config(main_config, create_cfg=create_config, auto=True)
    env = CliffWalkingEnv(cfg.env)
    env.enable_save_replay(replay_path=f'./{main_config.exp_name}/video')
    model = DQN(**cfg.policy.model)
    state_dict = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(state_dict['model'])
    policy = DQNPolicy(cfg.policy, model=model).eval_mode
    forward_fn = single_env_forward_wrapper(policy.forward)
    obs = env.reset()
    returns = 0.
    while True:
        action = forward_fn(obs)
        obs, rew, done, info = env.step(action)
        returns += rew
        if done:
            break
    print(f'Deploy is finished, final epsiode return is: {returns}')


if __name__ == "__main__":
    main(
        main_config=main_config,
        create_config=create_config,
        ckpt_path=f'./cliffwalking_dqn_seed0/ckpt/ckpt_best.pth.tar'
    )
