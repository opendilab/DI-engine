from easydict import EasyDict
import os
import gym
from ding.envs import BaseEnv, DingEnvWrapper
from ding.envs.env_wrappers import MaxAndSkipWrapper, WarpFrameWrapper, ScaledFloatFrameWrapper, FrameStackWrapper, \
    EvalEpisodeReturnEnv, TransposeWrapper, TimeLimitWrapper, FlatObsWrapper, GymToGymnasiumWrapper
from ding.policy import PPOFPolicy, A2CPolicy ,TD3Policy, DDPGPolicy, SACPolicy, DQNPolicy, IMPALAPolicy, C51Policy


def get_instance_config(env: str, algorithm: str) -> EasyDict:
    if algorithm == 'PPO':
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
        elif env == 'acrobot':
            cfg.learning_rate = 1e-4
            cfg.n_sample = 400
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
        elif env == 'minigrid_fourroom':
            cfg.n_sample = 3200
            cfg.batch_size = 320
            cfg.learning_rate = 3e-4
            cfg.epoch_per_collect = 10
            cfg.entropy_weight = 0.001
        elif env == 'metadrive':
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
        elif env in ['hopper']:
            cfg.action_space = "continuous"
            cfg.n_sample = 3200
            cfg.batch_size = 320
            cfg.epoch_per_collect = 10
            cfg.learning_rate = 3e-4
        else:
            raise KeyError("not supported env type: {}".format(env))
    elif algorithm == 'A2C':
        cfg = EasyDict({"policy": A2CPolicy.default_config()})
        if env == 'lunarlander_discrete':
            cfg.update(
                dict(
                    exp_name='LunarLander-v2-A2C',
                    env=dict(
                        collector_env_num=8,
                        evaluator_env_num=8,
                        env_id='LunarLander-v2',
                        n_evaluator_episode=8,
                        stop_value=240,
                    ),
                    policy=dict(
                        cuda=True,
                        model=dict(
                            obs_shape=8,
                            action_shape=4,
                        ),
                        learn=dict(
                            batch_size=160,
                            learning_rate=3e-4,
                            entropy_weight=0.001,
                            adv_norm=True,
                        ),
                        collect=dict(
                            n_sample=320,
                            discount_factor=0.99,
                            gae_lambda=0.95,
                        ),
                    ),
                    wandb_logger=dict(
                        gradient_logger=True,
                        video_logger=True,
                        plot_logger=True,
                        action_logger=True,
                        return_logger=False
                    ),
                )
            )
        else:
            raise KeyError("not supported env type: {}".format(env))
    elif algorithm == 'TD3':
        cfg = EasyDict({"policy": TD3Policy.default_config()})
        if env == 'hopper':
            cfg.update(
                dict(
                    exp_name='Hopper-v3-TD3',
                    seed=0,
                    env=dict(
                        env_id='Hopper-v3',
                        collector_env_num=8,
                        evaluator_env_num=8,
                        n_evaluator_episode=8,
                        stop_value=6000,
                    ),
                    policy=dict(
                        cuda=True,
                        random_collect_size=25000,
                        model=dict(
                            obs_shape=11,
                            action_shape=3,
                            actor_head_hidden_size=256,
                            critic_head_hidden_size=256,
                            action_space='regression',
                        ),
                        collect=dict(n_sample=1, ),
                        other=dict(replay_buffer=dict(replay_buffer_size=1000000, ), ),
                    ),
                    wandb_logger=dict(
                        gradient_logger=True,
                        video_logger=True,
                        plot_logger=True,
                        action_logger=True,
                        return_logger=False
                    ),
                )
            )
        elif env == 'HalfCheetah':
            cfg.update(
                dict(
                    exp_name='HalfCheetah-v3-TD3',
                    seed=0,
                    env=dict(
                        env_id='HalfCheetah-v3',
                        norm_obs=dict(use_norm=False, ),
                        norm_reward=dict(use_norm=False, ),
                        collector_env_num=1,
                        evaluator_env_num=8,
                        n_evaluator_episode=8,
                        stop_value=11000,
                    ),
                    policy=dict(
                        cuda=True,
                        random_collect_size=25000,
                        model=dict(
                            obs_shape=17,
                            action_shape=6,
                            twin_critic=True,
                            actor_head_hidden_size=256,
                            critic_head_hidden_size=256,
                            action_space='regression',
                        ),
                        learn=dict(
                            update_per_collect=1,
                            batch_size=256,
                            learning_rate_actor=1e-3,
                            learning_rate_critic=1e-3,
                            ignore_done=True,
                            target_theta=0.005,
                            discount_factor=0.99,
                            actor_update_freq=2,
                            noise=True,
                            noise_sigma=0.2,
                            noise_range=dict(
                                min=-0.5,
                                max=0.5,
                            ),
                        ),
                        collect=dict(
                            n_sample=1,
                            unroll_len=1,
                            noise_sigma=0.1,
                        ),
                        other=dict(replay_buffer=dict(replay_buffer_size=1000000, ), ),
                    ),
                    wandb_logger=dict(
                        gradient_logger=True,
                        video_logger=True,
                        plot_logger=True,
                        action_logger=True,
                        return_logger=False
                    ),
                )
            )
        elif env == 'Walker2d':
            cfg.update(
                dict(
                    exp_name='Walker2d-v3-TD3',
                    seed=0,
                    env=dict(
                        env_id='Walker2d-v3',
                        norm_obs=dict(use_norm=False, ),
                        norm_reward=dict(use_norm=False, ),
                        collector_env_num=1,
                        evaluator_env_num=8,
                        n_evaluator_episode=8,
                        stop_value=6000,
                    ),
                    policy=dict(
                        cuda=True,
                        random_collect_size=25000,
                        model=dict(
                            obs_shape=17,
                            action_shape=6,
                            twin_critic=True,
                            actor_head_hidden_size=256,
                            critic_head_hidden_size=256,
                            action_space='regression',
                        ),
                        learn=dict(
                            update_per_collect=1,
                            batch_size=256,
                            learning_rate_actor=1e-3,
                            learning_rate_critic=1e-3,
                            ignore_done=False,
                            target_theta=0.005,
                            discount_factor=0.99,
                            actor_update_freq=2,
                            noise=True,
                            noise_sigma=0.2,
                            noise_range=dict(
                                min=-0.5,
                                max=0.5,
                            ),
                        ),
                        collect=dict(
                            n_sample=1,
                            unroll_len=1,
                            noise_sigma=0.1,
                        ),
                        other=dict(replay_buffer=dict(replay_buffer_size=1000000, ), ),
                    ),
                    wandb_logger=dict(
                        gradient_logger=True,
                        video_logger=True,
                        plot_logger=True,
                        action_logger=True,
                        return_logger=False
                    ),
                )
            )
        elif env == 'lunarlander_continuous':
            cfg.update(
                dict(
                    exp_name='LunarLanderContinuous-V2-TD3',
                    seed=0,
                    env=dict(
                        env_id='LunarLanderContinuous-v2',
                        collector_env_num=4,
                        evaluator_env_num=8,
                        n_evaluator_episode=8,
                        act_scale=True,
                        stop_value=240,
                    ),
                    policy=dict(
                        cuda=True,
                        random_collect_size=10000,
                        model=dict(
                            obs_shape=8,
                            action_shape=2,
                            action_space='regression',
                        ),
                        learn=dict(
                            update_per_collect=256,
                            batch_size=256,
                            learning_rate_actor=3e-4,
                            learning_rate_critic=1e-3,
                            noise=True,
                            noise_sigma=0.1,
                            noise_range=dict(
                                min=-0.5,
                                max=0.5,
                            ),
                        ),
                        collect=dict(
                            n_sample=256,
                            noise_sigma=0.1,
                        ),
                        eval=dict(evaluator=dict(eval_freq=1000, ), ),
                        other=dict(replay_buffer=dict(replay_buffer_size=100000, ), ),
                    ),
                    wandb_logger=dict(
                        gradient_logger=True,
                        video_logger=True,
                        plot_logger=True,
                        action_logger=True,
                        return_logger=False
                    ),
                )
            )
        else:
            raise KeyError("not supported env type: {}".format(env))
    elif algorithm == 'DDPG':
        cfg = EasyDict({"policy": DDPGPolicy.default_config()})
        if env == 'hopper':
            cfg.update(
                dict(
                    exp_name='Hopper-v3-DDPG',
                    seed=0,
                    env=dict(
                        env_id='Hopper-v3',
                        norm_obs=dict(use_norm=False, ),
                        norm_reward=dict(use_norm=False, ),
                        collector_env_num=1,
                        evaluator_env_num=8,
                        n_evaluator_episode=8,
                        stop_value=6000,
                    ),
                    policy=dict(
                        cuda=True,
                        random_collect_size=25000,
                        model=dict(
                            obs_shape=11,
                            action_shape=3,
                            twin_critic=False,
                            actor_head_hidden_size=256,
                            critic_head_hidden_size=256,
                            action_space='regression',
                        ),
                        learn=dict(
                            update_per_collect=1,
                            batch_size=256,
                            learning_rate_actor=1e-3,
                            learning_rate_critic=1e-3,
                            ignore_done=False,
                            target_theta=0.005,
                            discount_factor=0.99,
                            actor_update_freq=1,
                            noise=False,
                        ),
                        collect=dict(
                            n_sample=1,
                            unroll_len=1,
                            noise_sigma=0.1,
                        ),
                        other=dict(replay_buffer=dict(replay_buffer_size=1000000, ), ),
                    )
                )
            )
        elif env == 'HalfCheetah':
            cfg.update(
                dict(
                    exp_name='HalfCheetah-v3-DDPG',
                    seed=0,
                    env=dict(
                        env_id='HalfCheetah-v3',
                        norm_obs=dict(use_norm=False, ),
                        norm_reward=dict(use_norm=False, ),
                        collector_env_num=1,
                        evaluator_env_num=8,
                        n_evaluator_episode=8,
                        stop_value=11000,
                    ),
                    policy=dict(
                        cuda=True,
                        random_collect_size=25000,
                        model=dict(
                            obs_shape=17,
                            action_shape=6,
                            twin_critic=False,
                            actor_head_hidden_size=256,
                            critic_head_hidden_size=256,
                            action_space='regression',
                        ),
                        learn=dict(
                            update_per_collect=1,
                            batch_size=256,
                            learning_rate_actor=1e-3,
                            learning_rate_critic=1e-3,
                            ignore_done=True,
                            target_theta=0.005,
                            discount_factor=0.99,
                            actor_update_freq=1,
                            noise=False,
                        ),
                        collect=dict(
                            n_sample=1,
                            unroll_len=1,
                            noise_sigma=0.1,
                        ),
                        other=dict(replay_buffer=dict(replay_buffer_size=1000000, ), ),
                    ),
                    wandb_logger=dict(
                        gradient_logger=True,
                        video_logger=True,
                        plot_logger=True,
                        action_logger=True,
                        return_logger=False
                    ),
                )
            )
        elif env == 'Walker2d':
            cfg.update(
                dict(
                    exp_name='Walker2d-v3-DDPG',
                    seed=0,
                    env=dict(
                        env_id='Walker2d-v3',
                        norm_obs=dict(use_norm=False, ),
                        norm_reward=dict(use_norm=False, ),
                        collector_env_num=1,
                        evaluator_env_num=8,
                        n_evaluator_episode=8,
                        stop_value=6000,
                    ),
                    policy=dict(
                        cuda=True,
                        random_collect_size=25000,
                        model=dict(
                            obs_shape=17,
                            action_shape=6,
                            twin_critic=False,
                            actor_head_hidden_size=256,
                            critic_head_hidden_size=256,
                            action_space='regression',
                        ),
                        learn=dict(
                            update_per_collect=1,
                            batch_size=256,
                            learning_rate_actor=1e-3,
                            learning_rate_critic=1e-3,
                            ignore_done=False,
                            target_theta=0.005,
                            discount_factor=0.99,
                            actor_update_freq=1,
                            noise=False,
                        ),
                        collect=dict(
                            n_sample=1,
                            unroll_len=1,
                            noise_sigma=0.1,
                        ),
                        other=dict(replay_buffer=dict(replay_buffer_size=1000000, ), ),
                    ),
                    wandb_logger=dict(
                        gradient_logger=True,
                        video_logger=True,
                        plot_logger=True,
                        action_logger=True,
                        return_logger=False
                    ),
                )
            )
        elif env == 'lunarlander_continuous':
            cfg.update(
                dict(
                    exp_name='LunarLanderContinuous-V2-DDPG',
                    seed=0,
                    env=dict(
                        env_id='LunarLanderContinuous-v2',
                        collector_env_num=8,
                        evaluator_env_num=8,
                        n_evaluator_episode=8,
                        act_scale=True,
                        stop_value=240,
                    ),
                    policy=dict(
                        cuda=True,
                        random_collect_size=0,
                        model=dict(
                            obs_shape=8,
                            action_shape=2,
                            twin_critic=True,
                            action_space='regression',
                        ),
                        learn=dict(
                            update_per_collect=2,
                            batch_size=128,
                            learning_rate_actor=0.001,
                            learning_rate_critic=0.001,
                            ignore_done=False,  # TODO(pu)
                            # (int) When critic network updates once, how many times will actor network update.
                            # Delayed Policy Updates in original TD3 paper(https://arxiv.org/pdf/1802.09477.pdf).
                            # Default 1 for DDPG, 2 for TD3.
                            actor_update_freq=1,
                            # (bool) Whether to add noise on target network's action.
                            # Target Policy Smoothing Regularization in original TD3 paper(https://arxiv.org/pdf/1802.09477.pdf).
                            # Default True for TD3, False for DDPG.
                            noise=False,
                            noise_sigma=0.1,
                            noise_range=dict(
                                min=-0.5,
                                max=0.5,
                            ),
                        ),
                        collect=dict(
                            n_sample=48,
                            noise_sigma=0.1,
                            collector=dict(collect_print_freq=1000, ),
                        ),
                        eval=dict(evaluator=dict(eval_freq=100, ), ),
                        other=dict(replay_buffer=dict(replay_buffer_size=20000, ), ),
                    ),
                    wandb_logger=dict(
                        gradient_logger=True,
                        video_logger=True,
                        plot_logger=True,
                        action_logger=True,
                        return_logger=False
                    ),
                )
            )
        else:
            raise KeyError("not supported env type: {}".format(env))
    elif algorithm == 'SAC':
        cfg = EasyDict({"policy": SACPolicy.default_config()})
        if env == 'hopper':
            cfg.update(
                dict(
                    exp_name='Hopper-v3-SAC',
                    seed=0,
                    env=dict(
                        env_id='Hopper-v3',
                        collector_env_num=8,
                        evaluator_env_num=8,
                        n_evaluator_episode=8,
                        stop_value=6000,
                    ),
                    policy=dict(
                        cuda=True,
                        random_collect_size=10000,
                        model=dict(
                            obs_shape=11,
                            action_shape=3,
                            action_space='reparameterization',
                            actor_head_hidden_size=256,
                            critic_head_hidden_size=256,
                        ),
                        learn=dict(
                            update_per_collect=1,
                            batch_size=256,
                            learning_rate_q=1e-3,
                            learning_rate_policy=1e-3,
                            reparameterization=True,
                            auto_alpha=False,
                        ),
                    ),
                    wandb_logger=dict(
                        gradient_logger=True,
                        video_logger=True,
                        plot_logger=True,
                        action_logger=True,
                        return_logger=False
                    ),
                )
            )
        elif env == 'HalfCheetah':
            cfg.update(
                dict(
                    exp_name='HalfCheetah-v3-SAC',
                    seed=0,
                    env=dict(
                        env_id='HalfCheetah-v3',
                        norm_obs=dict(use_norm=False, ),
                        norm_reward=dict(use_norm=False, ),
                        collector_env_num=1,
                        evaluator_env_num=8,
                        n_evaluator_episode=8,
                        stop_value=12000,
                    ),
                    policy=dict(
                        cuda=False,
                        random_collect_size=10000,
                        model=dict(
                            obs_shape=17,
                            action_shape=6,
                            twin_critic=True,
                            action_space='reparameterization',
                            actor_head_hidden_size=256,
                            critic_head_hidden_size=256,
                        ),
                        learn=dict(
                            update_per_collect=1,
                            batch_size=256,
                            learning_rate_q=1e-3,
                            learning_rate_policy=1e-3,
                            learning_rate_alpha=3e-4,
                            ignore_done=True,
                            target_theta=0.005,
                            discount_factor=0.99,
                            alpha=0.2,
                            reparameterization=True,
                            auto_alpha=False,
                        ),
                        collect=dict(
                            n_sample=1,
                            unroll_len=1,
                        ),
                        command=dict(),
                        eval=dict(),
                        other=dict(replay_buffer=dict(replay_buffer_size=1000000, ), ),
                    ),
                    wandb_logger=dict(
                        gradient_logger=True,
                        video_logger=True,
                        plot_logger=True,
                        action_logger=True,
                        return_logger=False
                    ),
                )
            )
        elif env == 'Walker2d':
            cfg.update(
                dict(
                    exp_name='Walker2d-v3-SAC',
                    seed=0,
                    env=dict(
                        env_id='Walker2d-v3',
                        norm_obs=dict(use_norm=False, ),
                        norm_reward=dict(use_norm=False, ),
                        collector_env_num=1,
                        evaluator_env_num=8,
                        n_evaluator_episode=8,
                        stop_value=6000,
                    ),
                    policy=dict(
                        cuda=True,
                        random_collect_size=10000,
                        model=dict(
                            obs_shape=17,
                            action_shape=6,
                            twin_critic=True,
                            action_space='reparameterization',
                            actor_head_hidden_size=256,
                            critic_head_hidden_size=256,
                        ),
                        learn=dict(
                            update_per_collect=1,
                            batch_size=256,
                            learning_rate_q=1e-3,
                            learning_rate_policy=1e-3,
                            learning_rate_alpha=3e-4,
                            ignore_done=False,
                            target_theta=0.005,
                            discount_factor=0.99,
                            alpha=0.2,
                            reparameterization=True,
                            auto_alpha=False,
                        ),
                        collect=dict(
                            n_sample=1,
                            unroll_len=1,
                        ),
                        command=dict(),
                        eval=dict(),
                        other=dict(replay_buffer=dict(replay_buffer_size=1000000, ), ),
                    ),
                    wandb_logger=dict(
                        gradient_logger=True,
                        video_logger=True,
                        plot_logger=True,
                        action_logger=True,
                        return_logger=False
                    ),
                )
            )
        elif env == 'lunarlander_continuous':
            cfg.update(
                dict(
                    exp_name='LunarLander-v2-SAC',
                    seed=0,
                    env=dict(
                        env_id='LunarLanderContinuous-v2',
                        collector_env_num=4,
                        evaluator_env_num=8,
                        act_scale=True,
                        n_evaluator_episode=8,
                        stop_value=240,
                    ),
                    policy=dict(
                        cuda=True,
                        random_collect_size=10000,
                        model=dict(
                            obs_shape=8,
                            action_shape=2,
                            action_space='reparameterization',
                            twin_critic=True,
                        ),
                        learn=dict(
                            update_per_collect=256,
                            batch_size=128,
                            learning_rate_q=1e-3,
                            learning_rate_policy=3e-4,
                            learning_rate_alpha=3e-4,
                            auto_alpha=True,
                        ),
                        collect=dict(n_sample=256, ),
                        eval=dict(evaluator=dict(eval_freq=1000, ), ),
                        other=dict(replay_buffer=dict(replay_buffer_size=int(1e5), ), ),
                    ),
                    wandb_logger=dict(
                        gradient_logger=True,
                        video_logger=True,
                        plot_logger=True,
                        action_logger=True,
                        return_logger=False
                    ),
                )
            )

            pass
        else:
            raise KeyError("not supported env type: {}".format(env))
    elif algorithm == 'DQN':
        cfg = EasyDict({"policy": DQNPolicy.default_config()})
        if env == 'lunarlander_discrete':
            cfg.update(
                dict(
                    exp_name='LunarLander-v2-DQN',
                    seed=0,
                    env=dict(
                        env_id='LunarLander-v2',
                        collector_env_num=8,
                        evaluator_env_num=8,
                        n_evaluator_episode=8,
                        stop_value=240,
                    ),
                    policy=dict(
                        cuda=True,
                        random_collect_size=25000,
                        discount_factor=0.99,
                        nstep=3,
                        learn=dict(
                            update_per_collect=10,
                            batch_size=64,
                            learning_rate=0.001,
                            # Frequency of target network update.
                            target_update_freq=100,
                        ),
                        model=dict(
                            obs_shape=8,
                            action_shape=4,
                            encoder_hidden_size_list=[512, 64],
                            # Whether to use dueling head.
                            dueling=True,
                        ),
                        collect=dict(
                            n_sample=64,
                            unroll_len=1,
                        ),
                        other=dict(
                            eps=dict(
                                type='exp',
                                start=0.95,
                                end=0.1,
                                decay=50000,
                            ),
                            replay_buffer=dict(replay_buffer_size=100000, )
                        ),
                    ),
                    wandb_logger=dict(
                        gradient_logger=True,
                        video_logger=True,
                        plot_logger=True,
                        action_logger=True,
                        return_logger=False
                    ),
                )
            )
        else:
            raise KeyError("not supported env type: {}".format(env))
    elif algorithm == 'C51':
        cfg = EasyDict({"policy": C51Policy.default_config()})
        if env == 'lunarlander_discrete':
            cfg.update(
                dict(
                    exp_name='lunarlander_c51',
                    seed=0,
                    env=dict(
                        collector_env_num=8,
                        evaluator_env_num=8,
                        env_id='LunarLander-v2',
                        n_evaluator_episode=8,
                        stop_value=200,
                    ),
                    policy=dict(
                        cuda=False,
                        model=dict(
                            obs_shape=8,
                            action_shape=4,
                            encoder_hidden_size_list=[512, 64],
                            v_min=-30,
                            v_max=30,
                            n_atom=51,
                        ),
                        discount_factor=0.99,
                        nstep=3,
                        learn=dict(
                            update_per_collect=10,
                            batch_size=64,
                            learning_rate=0.001,
                            target_update_freq=100,
                        ),
                        collect=dict(
                            n_sample=64,
                            unroll_len=1,
                        ),
                        other=dict(
                            eps=dict(
                                type='exp',
                                start=0.95,
                                end=0.1,
                                decay=50000,
                            ), replay_buffer=dict(replay_buffer_size=100000, )
                        ),
                    ),
                    wandb_logger=dict(
                        gradient_logger=True,
                        video_logger=True,
                        plot_logger=True,
                        action_logger=True,
                        return_logger=False
                    ),
                )
            )
        else:
            raise KeyError("not supported env type: {}".format(env))
    else:
        raise KeyError("not supported algorithm type: {}".format(algorithm))

    return cfg


def get_instance_env(env: str) -> BaseEnv:
    if env == 'lunarlander_discrete':
        return DingEnvWrapper(gym.make('LunarLander-v2'))
    elif env == 'lunarlander_continuous':
        return DingEnvWrapper(gym.make('LunarLander-v2', continuous=True))
    elif env == 'bipedalwalker':
        return DingEnvWrapper(gym.make('BipedalWalker-v3'), cfg={'act_scale': True})
    elif env == 'acrobot':
        return DingEnvWrapper(gym.make('Acrobot-v1'))
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
    elif env == 'hopper':
        cfg = EasyDict(
            env_id='Hopper-v3',
            env_wrapper='mujoco_default',
        )
        return DingEnvWrapper(gym.make('Hopper-v3'), cfg=cfg)
    elif env == 'HalfCheetah':
        cfg = EasyDict(
            env_id='HalfCheetah-v3',
            env_wrapper='mujoco_default',
        )
        return DingEnvWrapper(gym.make('HalfCheetah-v3'), cfg=cfg)
    elif env == 'Walker2d':
        cfg = EasyDict(
            env_id='Walker2d-v3',
            env_wrapper='mujoco_default',
        )
        return DingEnvWrapper(gym.make('Walker2d-v3'), cfg=cfg)
    elif env == "SpaceInvaders":
        cfg = EasyDict({
            'env_id': "SpaceInvaders-v4",
            'env_wrapper': 'atari_default',
        })
        return DingEnvWrapper(gym.make("SpaceInvaders-v4"), cfg=cfg)
    elif env in ['atari_qbert', 'atari_kangaroo', 'atari_bowling', 'atari_breakout', 'atari_spaceinvader',
                 'atari_gopher']:
        from dizoo.atari.envs.atari_env import AtariEnv
        atari_env_list = {
            'atari_qbert': 'QbertNoFrameskip-v4',
            'atari_kangaroo': 'KangarooNoFrameskip-v4',
            'atari_bowling': 'BowlingNoFrameskip-v4',
            'atari_breakout': 'BreakoutNoFrameskip-v4',
            'atari_spaceinvader': 'SpaceInvadersNoFrameskip-v4',
            'atari_gopher': 'GopherNoFrameskip-v4'
        }
        cfg = EasyDict({
            'env_id': atari_env_list[env],
            'env_wrapper': 'atari_default',
        })
        ding_env_atari = DingEnvWrapper(gym.make(atari_env_list[env]), cfg=cfg)
        return ding_env_atari
    elif env == 'minigrid_fourroom':
        import gymnasium
        return DingEnvWrapper(
            gymnasium.make('MiniGrid-FourRooms-v0'),
            cfg={
                'env_wrapper': [
                    lambda env: GymToGymnasiumWrapper(env),
                    lambda env: FlatObsWrapper(env),
                    lambda env: TimeLimitWrapper(env, max_limit=300),
                    lambda env: EvalEpisodeReturnEnv(env),
                ]
            }
        )
    elif env == 'metadrive':
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
        raise KeyError("not supported env type: {}".format(env))


def get_hybrid_shape(action_space) -> EasyDict:
    return EasyDict({
        'action_type_shape': action_space[0].n,
        'action_args_shape': action_space[1].shape,
    })
