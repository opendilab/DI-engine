from easydict import EasyDict
import gym
from ding.envs import BaseEnv, DingEnvWrapper
from ding.policy import PPOFPolicy


def get_instance_config(env: str) -> EasyDict:
    cfg = PPOFPolicy.default_config()
    if env == 'lunarlander_discrete':
        cfg.n_sample = 400
    elif env == 'lunarlander_continuous':
        cfg.action_space = 'continuous'
        cfg.n_sample = 400
    elif env == 'rocket_landing':
        cfg.n_sample = 2048
        cfg.model = dict(
            encoder_hidden_size_list=[64, 64, 128],
            actor_head_hidden_size=128,
            critic_head_hidden_size=128,
        )
    elif env == 'hybrid_moving':
        cfg.action_space = 'hybrid'
        cfg.n_sample = 3200
        cfg.entropy_weight = 0.03
        cfg.batch_size = 320
        cfg.adv_norm = False
        cfg.model = dict(
            encoder_hidden_size_list=[256, 128, 64, 64],
            sigma_type='fixed',
            fixed_sigma_value=0.3,
            bound_type='tanh',
        )
    else:
        raise KeyError("not supported env type: {}".format(env))
    return cfg


def get_instance_env(env: str) -> BaseEnv:
    if env == 'lunarlander_discrete':
        return DingEnvWrapper(gym.make('LunarLander-v2'))
    elif env == 'lunarlander_continuous':
        return DingEnvWrapper(gym.make('LunarLander-v2', continuous=True))
    elif env == 'hybrid_moving':
        import gym_hybrid
        return DingEnvWrapper(gym.make('Moving-v0'))
    elif env == 'rocket_landing':
        from dizoo.rocket.envs import RocketEnv
        cfg = EasyDict({
            'task': 'landing',
            'max_steps': 800,
        })
        return RocketEnv(cfg)
    else:
        raise KeyError("not supported env type: {}".format(env))


def get_hybrid_shape(action_space) -> EasyDict:
    return EasyDict({
        'action_type_shape': action_space[0].n,
        'action_args_shape': action_space[1].shape,
    })
