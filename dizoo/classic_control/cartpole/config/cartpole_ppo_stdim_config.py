from easydict import EasyDict

collector_env_num = 8
evaluator_env_num = 5
cartpole_ppo_stdim_config = dict(
    exp_name='cartpole_onppo_stdim_seed0',
    env=dict(
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        stop_value=195,
    ),
    policy=dict(
        cuda=True,
        action_space='discrete',
        model=dict(
            obs_shape=4,
            action_shape=2,
            action_space='discrete',
            encoder_hidden_size_list=[64, 64, 128],
            critic_head_hidden_size=128,
            actor_head_hidden_size=128,
        ),
        aux_model=dict(
            encode_shape=64,
            heads=[1, 1],
            loss_type='infonce',
            temperature=1.0,
        ),
        # the weight of the auxiliary loss to the TD loss
        aux_loss_weight=0.003,
        learn=dict(
            epoch_per_collect=2,
            batch_size=64,
            learning_rate=0.001,
            value_weight=0.5,
            entropy_weight=0.01,
            clip_ratio=0.2,
            learner=dict(hook=dict(save_ckpt_after_iter=100)),
        ),
        collect=dict(
            n_sample=256,
            unroll_len=1,
            discount_factor=0.9,
            gae_lambda=0.95,
        ),
        eval=dict(evaluator=dict(eval_freq=100, ), ),
    ),
)
cartpole_ppo_stdim_config = EasyDict(cartpole_ppo_stdim_config)
main_config = cartpole_ppo_stdim_config
cartpole_ppo_stdim_create_config = dict(
    env=dict(
        type='cartpole',
        import_names=['dizoo.classic_control.cartpole.envs.cartpole_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(type='ppo_stdim'),
)
cartpole_ppo_stdim_create_config = EasyDict(cartpole_ppo_stdim_create_config)
create_config = cartpole_ppo_stdim_create_config

if __name__ == "__main__":
    # or you can enter `ding -m serial_onpolicy -c cartpole_ppo_stdim_config.py -s 0`
    from ding.entry import serial_pipeline_onpolicy
    serial_pipeline_onpolicy((main_config, create_config), seed=0)
