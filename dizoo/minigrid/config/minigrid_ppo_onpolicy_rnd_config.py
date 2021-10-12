from easydict import EasyDict
from ding.entry import serial_pipeline_reward_model_onpolicy
import torch
print(torch.__version__,torch.cuda.is_available())
collector_env_num=8
minigrid_ppo_rnd_config = dict(
    # exp_name='minigrid_empty8_ppo_onpolicy_rnd_debug',
    exp_name='minigrid_fourrooms_ppo_onpolicy_rnd_debug',
    env=dict(
        collector_env_num= collector_env_num,
        evaluator_env_num=5,
        n_evaluator_episode=5,
        # env_id='MiniGrid-Empty-8x8-v0',
        env_id='MiniGrid-FourRooms-v0',
        stop_value=0.96,
    ),
    reward_model=dict(
        intrinsic_reward_type='add',  # 'assign'
        learning_rate=0.001,
        obs_shape=2739,
        # batch_size=32,
        # update_per_collect=10,
        batch_size=64, #32,
        update_per_collect=8, #10
        clear_buffer_per_iters=10,
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
            critic_head_hidden_size=64, # default=64
            actor_head_hidden_size=64,
        ),
        learn=dict(
            epoch_per_collect=10,  # TODO(pu)
            update_per_collect=1,  # 4
            batch_size=64,
            learning_rate=0.0003,
            value_weight=0.5,
            entropy_weight=0.001, 
            clip_ratio=0.2,
            adv_norm=True,
            value_norm=True, 
        ),
        collect=dict(
            collector_env_num= collector_env_num,
            n_sample=int(64*collector_env_num), 
            #  self._traj_len  = max(1,64*8//8)=64 
            #    self._traj_len = max(
            #      self._unroll_len,
            #     self._default_n_sample // self._env_num + int(self._default_n_sample % self._env_num != 0)
            #  )
            unroll_len=1,
            discount_factor=0.999,
            gae_lambda=0.95,
        ),
    ),
)
minigrid_ppo_rnd_config = EasyDict(minigrid_ppo_rnd_config)
main_config = minigrid_ppo_rnd_config
minigrid_ppo_rnd_create_config = dict(
    env=dict(
        type='minigrid',
        import_names=['dizoo.minigrid.envs.minigrid_env'],
    ),
    env_manager=dict(type='base'),
    # env_manager=dict(type='subprocess'),
    # policy=dict(type='ppo_offpolicy'),
    policy=dict(type='ppo'),
    reward_model=dict(type='rnd'),
)
minigrid_ppo_rnd_create_config = EasyDict(minigrid_ppo_rnd_create_config)
create_config = minigrid_ppo_rnd_create_config

if __name__ == "__main__":
    serial_pipeline_reward_model_onpolicy([main_config, create_config], seed=0)
