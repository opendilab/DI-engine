from easydict import EasyDict

beergame_ppo_config = dict(
    exp_name='beergame_ppo_seed0',
    env=dict(
        collector_env_num=8,
        evaluator_env_num=8,
        # env_id='beergame-v2',
        n_evaluator_episode=8,
        stop_value=200,
        role=0,  # 0-3 : retailer, warehouse, distributor, manufacturer
        agent_type='bs',
        replay_path=None
        # type of co-player, 'bs'- base stock, 'Strm'- use Sterman formula to model typical human behavior.
    ),
    policy=dict(
        cuda=True,
        recompute_adv=True,
        action_space='discrete',
        model=dict(
            obs_shape=50,  # statedim * multPerdInpt= 5 * 10
            action_shape=5,  # the quantity relative to the arriving order
            action_space='discrete',
            encoder_hidden_size_list=[64, 64, 128],
            actor_head_hidden_size=128,
            critic_head_hidden_size=128,
        ),
        learn=dict(
            epoch_per_collect=10,
            batch_size=320,
            learning_rate=3e-4,
            value_weight=0.5,
            entropy_weight=0.001,
            clip_ratio=0.2,
            adv_norm=True,
            value_norm=True,
            # for onppo, when we recompute adv, we need the key done in data to split traj, so we must
            # use ignore_done=False here,
            # but when we add key traj_flag in data as the backup for key done, we could choose to use ignore_done=True
            # for halfcheetah, the length=1000
            ignore_done=True,
            grad_clip_type='clip_norm',
            grad_clip_value=0.5,
        ),
        collect=dict(
            n_episode=8,
            unroll_len=1,
            discount_factor=0.99,
            gae_lambda=0.95,
            collector=dict(
                get_train_sample=True,
                reward_shaping=True,  # whether use total reward to reshape reward
            ),
        ),
        eval=dict(evaluator=dict(eval_freq=500, )),
    ),
)
beergame_ppo_config = EasyDict(beergame_ppo_config)
main_config = beergame_ppo_config
beergame_ppo_create_config = dict(
    env=dict(
        type='beergame',
        import_names=['dizoo.beergame.envs.beergame_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(type='ppo'),
    collector=dict(
        type='episode',
        get_train_sample=True,
        reward_shaping=True,  # whether use total reward to reshape reward
    ),
)
beergame_ppo_create_config = EasyDict(beergame_ppo_create_config)
create_config = beergame_ppo_create_config

if __name__ == "__main__":
    # or you can enter `ding -m serial -c beergame_offppo_config.py -s 0`
    from ding.entry import serial_pipeline_onpolicy
    serial_pipeline_onpolicy([main_config, create_config], seed=0)
