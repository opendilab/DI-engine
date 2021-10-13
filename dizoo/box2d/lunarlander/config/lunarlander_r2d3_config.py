from easydict import EasyDict

from ding.entry import serial_pipeline_r2d3

collector_env_num = 8
evaluator_env_num = 5

"""agent config"""
lunarlander_r2d3_config = dict(
    exp_name='lunarlander_r2d3_debug',
    env=dict(
        # Whether to use shared memory. Only effective if "env_manager_type" is 'subprocess'
        manager=dict(shared_memory=True, force_reproducibility=True),
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=5,
        stop_value=200,
    ),
    policy=dict(
        cuda=False,
        on_policy=False,
        priority=True,
        model=dict(
            obs_shape=8,
            action_shape=4,
            encoder_hidden_size_list=[128, 128, 64],
        ),
        discount_factor=0.997,
        burnin_step=20,
        nstep=5,
        # (int) the whole sequence length to unroll the RNN network minus
        # the timesteps of burnin part,
        # i.e., <the whole sequence length> = <burnin_step> + <unroll_len>
        unroll_len=80,
        learn=dict(
            # according to the r2d3 paper, actor parameter update interval is 400
            # environment timesteps, and in per collect phase, we collect 32 sequence
            # samples, the length of each samlpe sequence is <burnin_step> + <unroll_len>,
            # which is 100 in our seeting, 32*100/400=8, so we set update_per_collect=8
            # in most environments
            value_rescale=False,
            update_per_collect=8,
            batch_size=32,
            learning_rate=0.0005,
            target_update_freq=2500,
            ###
            lambda1=1.0,  # n-step return
            lambda2=1.0,  # supervised loss
            lambda3=1e-5,  # L2
            margin_function=0.8,  # margin function in JE, here we implement this as a constant
            per_train_iter_k=10,  # TODO(pu)
        ),
        collect=dict(
            # NOTE it is important that don't include key n_sample here, to make sure self._traj_len=INF
            # Cut trajectories into pieces with length "unroll_len".
            # unroll_len=1,
            env_num=collector_env_num,
            # The hyperparameter pho, the demo ratio, control the propotion of data coming\
            # from expert demonstrations versus from the agent's own experience.
            pho=0.25,
        ),
        eval=dict(env_num=evaluator_env_num, ),
        other=dict(
            eps=dict(
                type='exp',
                start=0.95,
                end=0.1,
                decay=10000,
            ),
            replay_buffer=dict(replay_buffer_size=100000, alpha=0.9),  # priority exponent =0.9
        ),
    ),
)
lunarlander_r2d3_config = EasyDict(lunarlander_r2d3_config)
main_config = lunarlander_r2d3_config
lunarlander_r2d3_create_config = dict(
    env=dict(
        type='lunarlander',
        import_names=['dizoo.box2d.lunarlander.envs.lunarlander_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(type='r2d3'),
)
lunarlander_r2d3_create_config = EasyDict(lunarlander_r2d3_create_config)
create_config = lunarlander_r2d3_create_config

"""export config"""

expert_lunarlander_r2d3_config = dict(
    exp_name='lunarlander_r2d3_debug',
    env=dict(
        # Whether to use shared memory. Only effective if "env_manager_type" is 'subprocess'
        manager=dict(shared_memory=True, force_reproducibility=True),
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=5,
        stop_value=200,
    ),
    policy=dict(
        cuda=False,
        on_policy=False,
        priority=True,
        model=dict(
            obs_shape=8,
            action_shape=4,
            # encoder_hidden_size_list=[512, 64],  # dqn
            # encoder_hidden_size_list=[128],  # ppo
        ),
        discount_factor=0.997,
        burnin_step=20,
        nstep=5,
        learn=dict(
            expert_replay_buffer_size=10000,  # TODO(pu)
        ),
        collect=dict(
            # n_sample=32, # NOTE it is important that don't include key n_sample here, to make sure self._traj_len=INF
            # Users should add their own path here (path should lead to a well-trained model)
            demonstration_info_path='dizoo/box2d/lunarlander/config/path/ppo-off_iteration_12948.pth.tar',
            # Cut trajectories into pieces with length "unroll_len". should set as self._unroll_len_add_burnin_step of r2d2
            unroll_len=100,
            env_num=collector_env_num,
        ),
        eval=dict(env_num=evaluator_env_num, ),
        other=dict(
            replay_buffer=dict(replay_buffer_size=10000, alpha=0.9),  # 1000 TODO(pu)
        ),
    ),
)
expert_lunarlander_r2d3_config = EasyDict(expert_lunarlander_r2d3_config)
expert_main_config = expert_lunarlander_r2d3_config
expert_lunarlander_r2d3_create_config = dict(
    env=dict(
        type='lunarlander',
        import_names=['dizoo.box2d.lunarlander.envs.lunarlander_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(type='ppo_offpolicy_collect_traj'),
)
expert_lunarlander_r2d3_create_config = EasyDict(expert_lunarlander_r2d3_create_config)
expert_create_config = expert_lunarlander_r2d3_create_config

if __name__ == "__main__":
    serial_pipeline_r2d3([main_config, create_config], [expert_main_config, expert_create_config], seed=0)
