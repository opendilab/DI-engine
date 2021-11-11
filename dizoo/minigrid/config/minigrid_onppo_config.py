from easydict import EasyDict
from ding.entry import serial_pipeline_onpolicy
collector_env_num = 8
minigrid_ppo_config = dict(
    # exp_name="minigrid_empty8_onppo",
    # exp_name="minigrid_fourrooms_onppo",
    exp_name="minigrid_doorkey88_onppo_seed2",
    # exp_name="minigrid_doorkey_onppo",

    env=dict(
        collector_env_num=8,
        evaluator_env_num=5,
        # env_id='MiniGrid-Empty-8x8-v0',
        # env_id='MiniGrid-FourRooms-v0',
        env_id='MiniGrid-DoorKey-8x8-v0',
        # env_id='MiniGrid-DoorKey-16x16-v0',
        n_evaluator_episode=5,
        stop_value=0.96,
    ),
    policy=dict(
        recompute_adv=True,
        cuda=True,
        continuous=False,
        on_policy=True,
        model=dict(
            obs_shape=2739,
            action_shape=7,
            encoder_hidden_size_list=[256, 128, 64, 64],
        ),
        learn=dict(
            epoch_per_collect=10,
            update_per_collect=1,
            batch_size=320,
            learning_rate=3e-4,
            value_weight=0.5,
            entropy_weight=0.001,
            clip_ratio=0.2,
            adv_norm=True,
            value_norm=True, 
        ),
        collect=dict(
            collector_env_num=collector_env_num,
            # n_sample=int(64 * collector_env_num),
            n_sample=int(3200), 

            #  self._traj_len  = max(1,64*8//8)=64
            #    self._traj_len = max(
            #      self._unroll_len,
            #     self._default_n_sample // self._env_num + int(self._default_n_sample % self._env_num != 0)
            #  )
            unroll_len=1,
            discount_factor=0.99,
            gae_lambda=0.95,
        ),
    ),
)
minigrid_ppo_config = EasyDict(minigrid_ppo_config)
main_config = minigrid_ppo_config
minigrid_ppo_create_config = dict(
    env=dict(
        type='minigrid',
        import_names=['dizoo.minigrid.envs.minigrid_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(type='ppo'),
)
minigrid_ppo_create_config = EasyDict(minigrid_ppo_create_config)
create_config = minigrid_ppo_create_config

if __name__ == "__main__":
    serial_pipeline_onpolicy([main_config, create_config], seed=2)
