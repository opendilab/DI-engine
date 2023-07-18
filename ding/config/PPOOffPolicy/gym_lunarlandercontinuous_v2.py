from easydict import EasyDict

action_shape = 2
obs_shape = 8

cfg = dict(
    exp_name='LunarLanderContinuous-v2-PPOOffPolicy',
    seed=0,
    env=dict(
        env_id='LunarLanderContinuous-v2',
        collector_env_num=8,
        evaluator_env_num=4,
        n_evaluator_episode=4,
        stop_value=240,
        act_scale=True,
    ),
    policy=dict(
        cuda=True,
        action_space='general',
        model=dict(
            obs_shape=8,
            action_shape=action_shape,
            action_space='general',
            customized_model=True,
            actor=dict(
                model_type='GaussianTanh',
                model=dict(
                    mu_model=dict(
                        hidden_sizes=[obs_shape, 256, 256],
                        activation=['relu', 'relu', 'tanh'],
                        output_size=action_shape,
                        dropout=0,
                        layernorm=False,
                        final_activation='tanh',
                        scale=5.0,
                        shrink=0.01,
                    ),
                    cov=dict(
                        dim=action_shape,
                        functional=True,
                        random_init=False,
                        sigma_lambda=dict(
                            hidden_sizes=[obs_shape, 128],
                            activation='tanh',
                            output_size=action_shape,
                            dropout=0,
                            layernorm=False,
                            final_activation='tanh',
                            scale=5.0,
                            offset=-5.0,
                        ),
                        sigma_offdiag=dict(
                            hidden_sizes=[obs_shape, 128],
                            activation='tanh',
                            output_size=int(action_shape * (action_shape - 1) // 2),
                            dropout=0,
                            layernorm=False,
                        ),
                    ),
                ),
            ),
            critic=dict(
                model_num=1,
                model=dict(
                    hidden_sizes=[obs_shape, 512, 256],
                    activation=['relu', 'softplus', 'softplus'],
                    output_size=1,
                    dropout=0,
                    layernorm=False,
                ),
            ),
        ),
        learn=dict(
            update_per_collect=1,
            batch_size=512,
            learning_rate=3e-4,
            value_weight=0.5,
            entropy_weight=0.01,
            clip_ratio=0.05,
            nstep=1,
            nstep_return=False,
            adv_norm=False,
            value_norm=False,
            ppo_param_init=False,
            separate_optimizer=True,
            weight_decay=0.0,
        ),
        collect=dict(
            n_sample=512,
            unroll_len=1,
            discount_factor=0.999,
            gae_lambda=1.0,
        ),
        eval=dict(
            evaluator=dict(eval_freq=100, ),
            render=True,
        ),
        other=dict(replay_buffer=dict(replay_buffer_size=int(512), ), ),
    ),
    wandb_logger=dict(
        gradient_logger=True, video_logger=True, plot_logger=True, action_logger=False, return_logger=False
    ),
)

cfg = EasyDict(cfg)

import ding.envs.gym_env
from functools import partial
env = partial(ding.envs.gym_env.env, continuous=True)
