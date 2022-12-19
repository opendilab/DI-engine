import gym
import torch
from easydict import EasyDict
from ding.config import compile_config
from ding.envs import DingEnvWrapper
from ding.policy import DQNPolicy, single_env_forward_wrapper
from ding.model import DQN
#from dizoo.box2d.lunarlander.config.lunarlander_dqn_config import main_config, create_config
from lunarlander_dqn_config import main_config, create_config


def main(main_config: EasyDict, create_config: EasyDict, ckpt_path: str):
    main_config.exp_name = 'lunarlander_dqn_deploy'

    cfg = compile_config(main_config, create_cfg=create_config, auto=True)

    env = DingEnvWrapper(gym.make(cfg.env.env_id), EasyDict(env_wrapper='default'))
    env.enable_save_replay(replay_path='./lunarlander_dqn_deploy/video')

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
    main(main_config=main_config, create_config=create_config, ckpt_path='lunarlander_dqn_seed0/ckpt/final.pth.tar')