from easydict import EasyDict
from ding.utils import set_pkg_seed

obs_shape = 4
action_shape = 2
num_agents = 100
dim_spin = 2
agent_view_sight = 1

ising_mfq_config = dict(
    exp_name='ising_mfq_seed0',
    env=dict(
        collector_env_num=8,
        evaluator_env_num=8,
        n_evaluator_episode=8,
        num_agents=num_agents,
        dim_spin=dim_spin,
        agent_view_sight=agent_view_sight,
        manager=dict(shared_memory=False, ),
    ),
    policy=dict(
        cuda=True,
        priority=False,
        model=dict(
            obs_shape=obs_shape + action_shape,  # for we will concat the pre_action_prob into obs
            action_shape=action_shape,
            encoder_hidden_size_list=[128, 128, 512],
            init_bias=0,
        ),
        nstep=3,
        discount_factor=0.99,
        learn=dict(
            update_per_collect=10,
            batch_size=32,
            learning_rate=0.0001,
            target_update_freq=500,
        ),
        collect=dict(n_sample=96, ),
        eval=dict(evaluator=dict(eval_freq=1000, )),
        other=dict(
            eps=dict(
                type='exp',
                start=1.,
                end=0.05,
                decay=250000,
            ),
            replay_buffer=dict(replay_buffer_size=100000, ),
        ),
    ),
)
ising_mfq_config = EasyDict(ising_mfq_config)
main_config = ising_mfq_config
ising_mfq_create_config = dict(
    env=dict(
        type='ising_model',
        import_names=['dizoo.ising_env.envs.ising_model_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='dqn'),
)
ising_mfq_create_config = EasyDict(ising_mfq_create_config)
create_config = ising_mfq_create_config

if __name__ == '__main__':
    # or you can enter `ding -m serial -c ising_mfq_config.py -s 0`
    from ding.entry import serial_pipeline
    seed = 1
    serial_pipeline((main_config, create_config), seed=seed, max_env_step=5e4)
