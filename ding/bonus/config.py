from easydict import EasyDict
import os
import gym
from ding.envs import BaseEnv, DingEnvWrapper
from ding.envs.env_wrappers import MaxAndSkipWrapper, WarpFrameWrapper, ScaledFloatFrameWrapper, FrameStackWrapper, \
    EvalEpisodeReturnWrapper, TransposeWrapper, TimeLimitWrapper, FlatObsWrapper, GymToGymnasiumWrapper
from ding.policy import PPOFPolicy


def get_instance_config(env_id: str, algorithm: str) -> EasyDict:
    if algorithm == 'PPOF':
        cfg = PPOFPolicy.default_config()
        if env_id == 'LunarLander-v2':
            cfg.n_sample = 512
            cfg.value_norm = 'popart'
            cfg.entropy_weight = 1e-3
        elif env_id == 'LunarLanderContinuous-v2':
            cfg.action_space = 'continuous'
            cfg.n_sample = 400
        elif env_id == 'BipedalWalker-v3':
            cfg.learning_rate = 1e-3
            cfg.action_space = 'continuous'
            cfg.n_sample = 1024
        elif env_id == 'acrobot':
            cfg.learning_rate = 1e-4
            cfg.n_sample = 400
        elif env_id == 'rocket_landing':
            cfg.n_sample = 2048
            cfg.adv_norm = False
            cfg.model = dict(
                encoder_hidden_size_list=[64, 64, 128],
                actor_head_hidden_size=128,
                critic_head_hidden_size=128,
            )
        elif env_id == 'drone_fly':
            cfg.action_space = 'continuous'
            cfg.adv_norm = False
            cfg.epoch_per_collect = 5
            cfg.learning_rate = 5e-5
            cfg.n_sample = 640
        elif env_id == 'hybrid_moving':
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
        elif env_id == 'evogym_carrier':
            cfg.action_space = 'continuous'
            cfg.n_sample = 2048
            cfg.batch_size = 256
            cfg.epoch_per_collect = 10
            cfg.learning_rate = 3e-3
        elif env_id == 'mario':
            cfg.n_sample = 256
            cfg.batch_size = 64
            cfg.epoch_per_collect = 2
            cfg.learning_rate = 1e-3
            cfg.model = dict(
                encoder_hidden_size_list=[64, 64, 128],
                critic_head_hidden_size=128,
                actor_head_hidden_size=128,
            )
        elif env_id == 'di_sheep':
            cfg.n_sample = 3200
            cfg.batch_size = 320
            cfg.epoch_per_collect = 10
            cfg.learning_rate = 3e-4
            cfg.adv_norm = False
            cfg.entropy_weight = 0.001
        elif env_id == 'procgen_bigfish':
            cfg.n_sample = 16384
            cfg.batch_size = 16384
            cfg.epoch_per_collect = 10
            cfg.learning_rate = 5e-4
            cfg.model = dict(
                encoder_hidden_size_list=[64, 128, 256],
                critic_head_hidden_size=256,
                actor_head_hidden_size=256,
            )
        elif env_id in ['KangarooNoFrameskip-v4', 'BowlingNoFrameskip-v4']:
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
        elif env_id == 'PongNoFrameskip-v4':
            cfg.n_sample = 3200
            cfg.batch_size = 320
            cfg.epoch_per_collect = 10
            cfg.learning_rate = 3e-4
            cfg.model = dict(
                encoder_hidden_size_list=[64, 64, 128],
                actor_head_hidden_size=128,
                critic_head_hidden_size=128,
            )
        elif env_id == 'SpaceInvadersNoFrameskip-v4':
            cfg.n_sample = 320
            cfg.batch_size = 320
            cfg.epoch_per_collect = 1
            cfg.learning_rate = 1e-3
            cfg.entropy_weight = 0.01
            cfg.lr_scheduler = (2000, 0.1)
            cfg.model = dict(
                encoder_hidden_size_list=[64, 64, 128],
                actor_head_hidden_size=128,
                critic_head_hidden_size=128,
            )
        elif env_id == 'QbertNoFrameskip-v4':
            cfg.n_sample = 3200
            cfg.batch_size = 320
            cfg.epoch_per_collect = 10
            cfg.learning_rate = 5e-4
            cfg.lr_scheduler = (1000, 0.1)
            cfg.model = dict(
                encoder_hidden_size_list=[64, 64, 128],
                actor_head_hidden_size=128,
                critic_head_hidden_size=128,
            )
        elif env_id == 'minigrid_fourroom':
            cfg.n_sample = 3200
            cfg.batch_size = 320
            cfg.learning_rate = 3e-4
            cfg.epoch_per_collect = 10
            cfg.entropy_weight = 0.001
        elif env_id == 'metadrive':
            cfg.learning_rate = 3e-4
            cfg.action_space = 'continuous'
            cfg.entropy_weight = 0.001
            cfg.n_sample = 3000
            cfg.epoch_per_collect = 10
            cfg.learning_rate = 0.0001
            cfg.model = dict(
                encoder_hidden_size_list=[32, 64, 64, 128],
                actor_head_hidden_size=128,
                critic_head_hidden_size=128,
                critic_head_layer_num=2,
            )
        elif env_id == 'Hopper-v3':
            cfg.action_space = "continuous"
            cfg.n_sample = 3200
            cfg.batch_size = 320
            cfg.epoch_per_collect = 10
            cfg.learning_rate = 3e-4
        elif env_id == 'HalfCheetah-v3':
            cfg.action_space = "continuous"
            cfg.n_sample = 3200
            cfg.batch_size = 320
            cfg.epoch_per_collect = 10
            cfg.learning_rate = 3e-4
        elif env_id == 'Walker2d-v3':
            cfg.action_space = "continuous"
            cfg.n_sample = 3200
            cfg.batch_size = 320
            cfg.epoch_per_collect = 10
            cfg.learning_rate = 3e-4
        else:
            raise KeyError("not supported env type: {}".format(env_id))
    else:
        raise KeyError("not supported algorithm type: {}".format(algorithm))

    return cfg


def get_instance_env(env_id: str) -> BaseEnv:
    if env_id == 'LunarLander-v2':
        return DingEnvWrapper(gym.make('LunarLander-v2'))
    elif env_id == 'LunarLanderContinuous-v2':
        return DingEnvWrapper(gym.make('LunarLanderContinuous-v2', continuous=True))
    elif env_id == 'BipedalWalker-v3':
        return DingEnvWrapper(gym.make('BipedalWalker-v3'), cfg={'act_scale': True, 'rew_clip': True})
    elif env_id == 'Pendulum-v1':
        return DingEnvWrapper(gym.make('Pendulum-v1'), cfg={'act_scale': True})
    elif env_id == 'acrobot':
        return DingEnvWrapper(gym.make('Acrobot-v1'))
    elif env_id == 'rocket_landing':
        from dizoo.rocket.envs import RocketEnv
        cfg = EasyDict({
            'task': 'landing',
            'max_steps': 800,
        })
        return RocketEnv(cfg)
    elif env_id == 'drone_fly':
        from dizoo.gym_pybullet_drones.envs import GymPybulletDronesEnv
        cfg = EasyDict({
            'env_id': 'flythrugate-aviary-v0',
            'action_type': 'VEL',
        })
        return GymPybulletDronesEnv(cfg)
    elif env_id == 'hybrid_moving':
        import gym_hybrid
        return DingEnvWrapper(gym.make('Moving-v0'))
    elif env_id == 'evogym_carrier':
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
                    lambda env: EvalEpisodeReturnWrapper(env),
                ]
            }
        )
    elif env_id == 'mario':
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
                    lambda env: EvalEpisodeReturnWrapper(env),
                ]
            }
        )
    elif env_id == 'di_sheep':
        from sheep_env import SheepEnv
        return DingEnvWrapper(SheepEnv(level=9))
    elif env_id == 'procgen_bigfish':
        return DingEnvWrapper(
            gym.make('procgen:procgen-bigfish-v0', start_level=0, num_levels=1),
            cfg={
                'env_wrapper': [
                    lambda env: TransposeWrapper(env),
                    lambda env: ScaledFloatFrameWrapper(env),
                    lambda env: EvalEpisodeReturnWrapper(env),
                ]
            },
            seed_api=False,
        )
    elif env_id == 'Hopper-v3':
        cfg = EasyDict(
            env_id='Hopper-v3',
            env_wrapper='mujoco_default',
            act_scale=True,
            rew_clip=True,
        )
        return DingEnvWrapper(gym.make('Hopper-v3'), cfg=cfg)
    elif env_id == 'HalfCheetah-v3':
        cfg = EasyDict(
            env_id='HalfCheetah-v3',
            env_wrapper='mujoco_default',
            act_scale=True,
            rew_clip=True,
        )
        return DingEnvWrapper(gym.make('HalfCheetah-v3'), cfg=cfg)
    elif env_id == 'Walker2d-v3':
        cfg = EasyDict(
            env_id='Walker2d-v3',
            env_wrapper='mujoco_default',
            act_scale=True,
            rew_clip=True,
        )
        return DingEnvWrapper(gym.make('Walker2d-v3'), cfg=cfg)

    elif env_id in [
            'BowlingNoFrameskip-v4',
            'BreakoutNoFrameskip-v4',
            'GopherNoFrameskip-v4'
            'KangarooNoFrameskip-v4',
            'PongNoFrameskip-v4',
            'QbertNoFrameskip-v4',
            'SpaceInvadersNoFrameskip-v4',
    ]:

        cfg = EasyDict({
            'env_id': env_id,
            'env_wrapper': 'atari_default',
        })
        ding_env_atari = DingEnvWrapper(gym.make(env_id), cfg=cfg)
        return ding_env_atari
    elif env_id == 'minigrid_fourroom':
        import gymnasium
        return DingEnvWrapper(
            gymnasium.make('MiniGrid-FourRooms-v0'),
            cfg={
                'env_wrapper': [
                    lambda env: GymToGymnasiumWrapper(env),
                    lambda env: FlatObsWrapper(env),
                    lambda env: TimeLimitWrapper(env, max_limit=300),
                    lambda env: EvalEpisodeReturnWrapper(env),
                ]
            }
        )
    elif env_id == 'metadrive':
        from dizoo.metadrive.env.drive_env import MetaDrivePPOOriginEnv
        from dizoo.metadrive.env.drive_wrapper import DriveEnvWrapper
        cfg = dict(
            map='XSOS',
            horizon=4000,
            out_of_road_penalty=40.0,
            crash_vehicle_penalty=40.0,
            out_of_route_done=True,
        )
        cfg = EasyDict(cfg)
        return DriveEnvWrapper(MetaDrivePPOOriginEnv(cfg))
    else:
        raise KeyError("not supported env type: {}".format(env_id))


def get_hybrid_shape(action_space) -> EasyDict:
    return EasyDict({
        'action_type_shape': action_space[0].n,
        'action_args_shape': action_space[1].shape,
    })
