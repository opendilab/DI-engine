from email import policy
from easydict import EasyDict


# learn learner
# collect
# other replay buffer

# td step
bipedalwalker_sac_config = dict(
    exp_name='bipedalwalker_sac_dt_dataset_seed0',
    env=dict(
        env_id='BipedalWalker-v3',
        collector_env_num=8,
        evaluator_env_num=5,
        # act scale
        act_scale=True,
        n_evaluator_episode=5,
        stop_value=300,
        rew_clip=True,
        # save game play
        replay_path=None,
    ),
    policy=dict(
        cuda=True,
        priority=False,
        model=dict(
            obs_shape=8,
            action_shape=4,
            twin_critic=True,
            action_space='reparameterization',
            actor_head_hidden_size=128,
            critic_head_hidden_size=128,
        ),
        learn=dict(
            # NOTE
            learner=dict(
                train_iterations=int(1e9),
                dataloader=dict(
                    num_workers=0,
                ),
                log_policy=True,
                hook=dict(
                    load_ckpt_before_run='/mnt/nfs/wangzilin/bipedalwalker_sac_seed0/ckpt/ckpt_best.pth.tar',
                    log_show_after_iter=100,
                    save_ckpt_after_iter=10000,
                    save_ckpt_after_run=False,
                ),
                cfg_type='BaseLearnerDict',
                load_path='/mnt/nfs/wangzilin/bipedalwalker_sac_seed0/ckpt/ckpt_best.pth.tar',
            ),
            update_per_collect=1,
            batch_size=128,
            learning_rate_q=0.001,
            learning_rate_policy=0.001,
            learning_rate_alpha=0.0003,
            ignore_done=True,
            target_theta=0.005,
            discount_factor=0.99,
            auto_alpha=True,
            value_network=False,
        ),
        collect=dict(
            n_sample=128,
            unroll_len=1,
            data_type='naive',
            save_path='/home/wangzilin/projects/decision_transformer/DI-engine/dizoo/box2d/bipedalwalker/dt_data/data/sac_data_1000eps.pkl',
            data_path='/home/wangzilin/projects/decision_transformer/DI-engine/dizoo/box2d/bipedalwalker/dt_data/data/sac_data_10eps.pkl',
        ),
        other=dict(
            replay_buffer=dict(
                type='advanced',
                replay_buffer_size=1000,
                max_use=float('inf'),
                max_staleness=float('inf'),
                alpha=0.6,
                beta=0.4,
                anneal_step=int(1e5),
                enable_track_used_data=False,
                deepcopy=False,
                thruput_controller=dict(
                    push_sample_rate_limit=dict(
                        max=float('inf'),
                        min=0,
                    ),
                    window_seconds=30,
                    sample_min_limit_ratio=1,
                ),
                monitor=dict(
                    sample_data_attr=dict(
                        average_range=5,
                        print_freq=200,
                    ),
                    periodic_thruput=dict(
                        seconds=60,
                    ),
                ),
                cfg_type='AdvancedReplayBufferDict',
            ),
        ),
    ),
)
bipedalwalker_sac_config = EasyDict(bipedalwalker_sac_config)
main_config = bipedalwalker_sac_config
bipedalwalker_sac_create_config = dict(
    env=dict(
        type='bipedalwalker',
        import_names=['dizoo.box2d.bipedalwalker.envs.bipedalwalker_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='sac',
        import_names=['ding.policy.sac'],
    ),
    replay_buffer=dict(type='naive',),
)

bipedalwalker_sac_create_config = EasyDict(bipedalwalker_sac_create_config)
create_config = bipedalwalker_sac_create_config

if __name__ == "__main__":
    # or you can enter `ding -m serial -c bipedalwalker_sac_config.py -s 0`
    from ding.entry import serial_pipeline
    serial_pipeline([main_config, create_config], seed=0)