"""
Overview:
    Here is the behaviour cloning (BC) main entry for gfootball.
    We first collect demo data using rule model, then train the BC model
    using the demo data whose return is larger than 0,
    and (optional) test accuracy in train dataset and test dataset of the trained BC model
"""
from copy import deepcopy
import os
import torch
import logging
import test_accuracy
from ding.entry import serial_pipeline_bc, collect_episodic_demo_data, episode_to_transitions_filter, eval
from ding.config import read_config, compile_config
from ding.policy import create_policy
from dizoo.gfootball.entry.gfootball_bc_config import gfootball_bc_config, gfootball_bc_create_config
from dizoo.gfootball.model.q_network.football_q_network import FootballNaiveQ
from dizoo.gfootball.model.bots.rule_based_bot_model import FootballRuleBaseModel

path = os.path.abspath(__file__)
dir_path = os.path.dirname(path)
logging.basicConfig(level=logging.INFO)

# Note: in gfootball env, 3000 transitions = one episode,
# 3e5 transitions = 200 episode, the memory needs about 350G.
seed = 0
gfootball_bc_config.exp_name = 'gfootball_bc_rule_200ep_lt0_seed0'
demo_episodes = 200  # key hyper-parameter
data_path_episode = dir_path + f'/gfootball_rule_{demo_episodes}eps.pkl'
data_path_transitions_lt0 = dir_path + f'/gfootball_rule_{demo_episodes}eps_transitions_lt0.pkl'
"""
phase 1: collect demo data utilizing rule model
"""
input_cfg = [deepcopy(gfootball_bc_config), deepcopy(gfootball_bc_create_config)]
if isinstance(input_cfg, str):
    cfg, create_cfg = read_config(input_cfg)
else:
    cfg, create_cfg = input_cfg
cfg = compile_config(cfg, seed=seed, auto=True, create_cfg=create_cfg)

football_rule_base_model = FootballRuleBaseModel()
expert_policy = create_policy(cfg.policy, model=football_rule_base_model, enable_field=['learn', 'collect', 'eval'])

# collect rule/expert demo data
state_dict = expert_policy.collect_mode.state_dict()
collect_config = [deepcopy(gfootball_bc_config), deepcopy(gfootball_bc_create_config)]
eval_config = deepcopy(collect_config)

# eval demo model
# if save replay
# eval(eval_config, seed=seed, model=football_rule_base_model, replay_path=dir_path + f'/gfootball_rule_replay/')
# if not save replay
# eval(eval_config, seed=seed, model=football_rule_base_model, state_dict=state_dict)

# collect demo data
collect_episodic_demo_data(
    collect_config,
    seed=seed,
    expert_data_path=data_path_episode,
    collect_count=demo_episodes,
    model=football_rule_base_model,
    state_dict=state_dict
)
# Note: only use the episode whose return is larger than 0 as demo data
episode_to_transitions_filter(
    data_path=data_path_episode, expert_data_path=data_path_transitions_lt0, nstep=1, min_episode_return=1
)
"""
phase 2: BC training
"""
bc_config = [deepcopy(gfootball_bc_config), deepcopy(gfootball_bc_create_config)]
bc_config[0].policy.learn.train_epoch = 1000  # key hyper-parameter
football_naive_q = FootballNaiveQ()

_, converge_stop_flag = serial_pipeline_bc(
    bc_config, seed=seed, data_path=data_path_transitions_lt0, model=football_naive_q
)

if bc_config[0].policy.show_train_test_accuracy:
    """
    phase 3: test accuracy in train dataset and test dataset
    """
    bc_model_path = bc_config[0].policy.bc_model_path

    # load trained model
    bc_config[0].policy.learn.batch_size = int(3000)  # the total dataset
    state_dict = torch.load(bc_model_path)
    football_naive_q.load_state_dict(state_dict['model'])
    policy = create_policy(cfg.policy, model=football_naive_q, enable_field=['eval'])

    # calculate accuracy in train dataset
    print('==' * 10)
    print('calculate accuracy in train dataset')
    print('==' * 10)
    # Users should add their own bc train_data_path here. Absolute path is recommended.
    train_data_path = dir_path + f'/gfootball_rule_100eps_transitions_lt0_train.pkl'
    test_accuracy.test_accuracy_in_dataset(train_data_path, cfg.policy.learn.batch_size, policy)

    # calculate accuracy in test dataset
    print('==' * 10)
    print('calculate accuracy in test dataset')
    print('==' * 10)
    # Users should add their own bc test_data_path here. Absolute path is recommended.
    test_data_path = dir_path + f'/gfootball_rule_50eps_transitions_lt0_test.pkl'
    test_accuracy.test_accuracy_in_dataset(test_data_path, cfg.policy.learn.batch_size, policy)
