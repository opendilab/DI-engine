from easydict import EasyDict
import os
import gym
from ding.envs import BaseEnv, DingEnvWrapper
from ding.envs.env_wrappers import MaxAndSkipWrapper, WarpFrameWrapper, ScaledFloatFrameWrapper, FrameStackWrapper, \
    EvalEpisodeReturnEnv, TransposeWrapper, TimeLimitWrapper
from ding.policy import PPOFPolicy


def get_instance_config(env: str) -> EasyDict:
    cfg = PPOFPolicy.default_config()
    if env == 'lunarlander_discrete':
        cfg.n_sample = 400
    elif env == 'lunarlander_continuous':
        cfg.action_space = 'continuous'
        cfg.n_sample = 400
    elif env == 'bipedalwalker':
        cfg.learning_rate = 1e-3
        cfg.action_space = 'continuous'
        cfg.n_sample = 1024
    elif env == 'rocket_landing':
        cfg.n_sample = 2048
        cfg.adv_norm = False
        cfg.model = dict(
            encoder_hidden_size_list=[64, 64, 128],
            actor_head_hidden_size=128,
            critic_head_hidden_size=128,
        )
    elif env == 'drone_fly':
        cfg.action_space = 'continuous'
        cfg.adv_norm = False
        cfg.epoch_per_collect = 5
        cfg.learning_rate = 5e-5
        cfg.n_sample = 640
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
    elif env == 'evogym_carrier':
        cfg.action_space = 'continuous'
        cfg.n_sample = 2048
        cfg.batch_size = 256
        cfg.epoch_per_collect = 10
        cfg.learning_rate = 3e-3
    elif env == 'mario':
        cfg.n_sample = 256
        cfg.batch_size = 64
        cfg.epoch_per_collect = 2
        cfg.learning_rate = 1e-3
        cfg.model = dict(
            encoder_hidden_size_list=[64, 64, 128],
            critic_head_hidden_size=128,
            actor_head_hidden_size=128,
        )
    elif env == 'di_sheep':
        cfg.n_sample = 3200
        cfg.batch_size = 320
        cfg.epoch_per_collect = 10
        cfg.learning_rate = 3e-4
        cfg.adv_norm = False
        cfg.entropy_weight = 0.001
    elif env == 'procgen_bigfish':
        cfg.n_sample = 16384
        cfg.batch_size = 16384
        cfg.epoch_per_collect = 10
        cfg.learning_rate = 5e-4
        cfg.model = dict(
            encoder_hidden_size_list=[64, 128, 256],
            critic_head_hidden_size=256,
            actor_head_hidden_size=256,
        )
    elif env in ['atari_qbert', 'atari_kangaroo', 'atari_bowling']:
        cfg.n_sample = 1024
        cfg.batch_size = 128
        cfg.epoch_per_collect = 10
        cfg.learning_rate = 0.0001
        cfg.model = dict(
            encoder_hidden_size_list=[32, 64, 64, 128],
            actor_head_hidden_size=128,
            critic_head_hidden_size=128,
            critic_head_layer_num=2,
        )
    else:
        raise KeyError("not supported env type: {}".format(env))
    return cfg


def get_instance_env(env: str) -> BaseEnv:
    if env == 'lunarlander_discrete':
        return DingEnvWrapper(gym.make('LunarLander-v2'))
    elif env == 'lunarlander_continuous':
        return DingEnvWrapper(gym.make('LunarLander-v2', continuous=True))
    elif env == 'bipedalwalker':
        return DingEnvWrapper(gym.make('BipedalWalker-v3'), cfg={'act_scale': True})
    elif env == 'rocket_landing':
        from dizoo.rocket.envs import RocketEnv
        cfg = EasyDict({
            'task': 'landing',
            'max_steps': 800,
        })
        return RocketEnv(cfg)
    elif env == 'drone_fly':
        from dizoo.gym_pybullet_drones.envs import GymPybulletDronesEnv
        cfg = EasyDict({
            'env_id': 'flythrugate-aviary-v0',
            'action_type': 'VEL',
        })
        return GymPybulletDronesEnv(cfg)
    elif env == 'hybrid_moving':
        import gym_hybrid
        return DingEnvWrapper(gym.make('Moving-v0'))
    elif env == 'evogym_carrier':
        import evogym.envs
        from evogym import sample_robot, WorldObject
        path = os.path.join(os.path.dirname(__file__), '../../dizoo/evogym/envs/world_data/carry_bot.json')
        robot_object = WorldObject.from_json(path)
        body = robot_object.get_structure()
        return DingEnvWrapper(
            gym.make('Carrier-v0', body=body),
            cfg={
                'env_wrapper': [
                    lambda env: TimeLimitWrapper(env, max_limit=300),
                    lambda env: EvalEpisodeReturnEnv(env),
                ]
            }
        )
    elif env == 'mario':
        import gym_super_mario_bros
        from nes_py.wrappers import JoypadSpace
        return DingEnvWrapper(
            JoypadSpace(gym_super_mario_bros.make("SuperMarioBros-1-1-v1"), [["right"], ["right", "A"]]),
            cfg={
                'env_wrapper': [
                    lambda env: MaxAndSkipWrapper(env, skip=4),
                    lambda env: WarpFrameWrapper(env, size=84),
                    lambda env: ScaledFloatFrameWrapper(env),
                    lambda env: FrameStackWrapper(env, n_frames=4),
                    lambda env: TimeLimitWrapper(env, max_limit=200),
                    lambda env: EvalEpisodeReturnEnv(env),
                ]
            }
        )
    elif env == 'di_sheep':
        from sheep_env import SheepEnv
        return DingEnvWrapper(SheepEnv(level=9))
    elif env == 'procgen_bigfish':
        return DingEnvWrapper(
            gym.make('procgen:procgen-bigfish-v0', start_level=0, num_levels=1),
            cfg={
                'env_wrapper': [
                    lambda env: TransposeWrapper(env),
                    lambda env: ScaledFloatFrameWrapper(env),
                    lambda env: EvalEpisodeReturnEnv(env),
                ]
            },
            seed_api=False,
        )
    elif env in ['atari_qbert', 'atari_kangaroo', 'atari_bowling']:
        from dizoo.atari.envs.atari_env import AtariEnv
        atari_env_list = {'atari_qbert': 'QbertNoFrameskip-v4', 'atari_kangaroo': 'KangarooNoFrameskip-v4',
                          'atari_bowling': 'BowlingNoFrameskip-v4'}
        cfg = EasyDict({
            'env_id': atari_env_list[env],
            'env_wrapper': 'atari_default',
        })
        ding_env_atari = DingEnvWrapper(gym.make(atari_env_list[env]), cfg=cfg)
        ding_env_atari.enable_save_replay(env + '_log/')
        obs = ding_env_atari.reset()
        return ding_env_atari
    else:
        raise KeyError("not supported env type: {}".format(env))


def get_hybrid_shape(action_space) -> EasyDict:
    return EasyDict({
        'action_type_shape': action_space[0].n,
        'action_args_shape': action_space[1].shape,
    })
