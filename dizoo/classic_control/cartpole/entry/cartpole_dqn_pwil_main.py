from easydict import EasyDict
from copy import deepcopy

from dizoo.classic_control.cartpole.config.cartpole_ppo_offpolicy_config import cartpole_ppo_offpolicy_config, cartpole_ppo_offpolicy_create_config  # noqa
from dizoo.classic_control.cartpole.config.cartpole_dqn_config import cartpole_dqn_config, cartpole_dqn_create_config
from ding.entry import serial_pipeline, collect_demo_data, serial_pipeline_reward_model_offpolicy


def cartpole_dqn_pwil_main():
    reward_model_config = {
        'type': 'pwil',
        's_size': 4,
        'a_size': 2,
        'sample_size': 500,
    }

    # train a expert policy (PPO offpolicy)
    reward_model_config = EasyDict(reward_model_config)
    config = deepcopy(cartpole_ppo_offpolicy_config), deepcopy(cartpole_ppo_offpolicy_create_config)
    expert_policy = serial_pipeline(config, seed=0)

    # (optional) collect expert demo data
    collect_count = 10000
    expert_data_path = 'expert_data.pkl'
    state_dict = expert_policy.collect_mode.state_dict()
    config = deepcopy(cartpole_ppo_offpolicy_config), deepcopy(cartpole_ppo_offpolicy_create_config)
    collect_demo_data(
        config, seed=0, state_dict=state_dict, expert_data_path=expert_data_path, collect_count=collect_count
    )

    # irl + rl training
    cp_cartpole_dqn_config = deepcopy(cartpole_dqn_config)
    cp_cartpole_dqn_create_config = deepcopy(cartpole_dqn_create_config)
    cp_cartpole_dqn_create_config.reward_model = dict(type=reward_model_config.type)
    reward_model_config['expert_data_path'] = expert_data_path
    cp_cartpole_dqn_config.exp_name = 'cartpole_dqn_pwil'
    cp_cartpole_dqn_config.reward_model = reward_model_config
    cp_cartpole_dqn_config.policy.collect.n_sample = 128

    serial_pipeline_reward_model_offpolicy((cp_cartpole_dqn_config, cp_cartpole_dqn_create_config), seed=0)


if __name__ == "__main__":
    cartpole_dqn_pwil_main()
