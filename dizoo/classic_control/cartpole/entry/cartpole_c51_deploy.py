import gym
import torch
from easydict import EasyDict
from ding.config import compile_config
from ding.envs import DingEnvWrapper
from ding.policy import C51Policy, single_env_forward_wrapper
from ding.model import C51DQN
from dizoo.classic_control.cartpole.config.cartpole_c51_config import cartpole_c51_config, cartpole_c51_create_config


def main(main_config: EasyDict, create_config: EasyDict, ckpt_path: str):
    main_config.exp_name = 'cartpole_c51_deploy'
    cfg = compile_config(main_config, create_cfg=create_config, auto=True)
    env = DingEnvWrapper(gym.make('CartPole-v0'), EasyDict(env_wrapper='default'))
    model = C51DQN(**cfg.policy.model)
    state_dict = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(state_dict['model'])
    policy = C51Policy(cfg.policy, model=model).eval_mode
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
    main(cartpole_c51_config, cartpole_c51_create_config, 'cartpole_c51_seed0/ckpt/ckpt_best.pth.tar')
