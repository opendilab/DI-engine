from copy import deepcopy
import os
import torch

path = os.path.abspath(__file__)
dir_path = os.path.dirname(path)

from ding.entry import serial_pipeline_bc, collect_demo_data, collect_episodic_demo_data, episode_to_transitions, \
    episode_to_transitions_filter, eval
from ding.config import read_config, compile_config
from ding.policy import create_policy
from dizoo.gfootball.entry.gfootball_il_config import gfootball_il_main_config, gfootball_il_create_config
from dizoo.gfootball.model.q_network.football_q_network import FootballNaiveQ
from dizoo.gfootball.model.bots.kaggle_5th_place_model import FootballKaggle5thPlaceModel

seed = 0
gfootball_il_main_config.exp_name = 'data_gfootball/gfootball_il_kaggle5th_seed0'
# in gfootball env: 3000 transitions = one episode
# 3e5 transitions = 100 episode, The memory needs about 180G
demo_transitions = int(3e5)  # key hyper-parameter
data_path_transitions = dir_path + f'/gfootball_kaggle5th_{demo_transitions}-demo-transitions.pkl'

"""
phase 1: train/obtain expert policy
"""
train_config = [deepcopy(gfootball_il_main_config), deepcopy(gfootball_il_create_config)]
input_cfg = train_config
if isinstance(input_cfg, str):
    cfg, create_cfg = read_config(input_cfg)
else:
    cfg, create_cfg = input_cfg
create_cfg.policy.type = create_cfg.policy.type + '_command'
env_fn = None
cfg = compile_config(cfg, seed=seed, env=env_fn, auto=True, create_cfg=create_cfg, save_cfg=True)

football_kaggle_5th_place_model = FootballKaggle5thPlaceModel()
expert_policy = create_policy(cfg.policy, model=football_kaggle_5th_place_model,
                              enable_field=['learn', 'collect', 'eval', 'command'])

# collect expert demo data
state_dict = expert_policy.collect_mode.state_dict()
collect_config = [deepcopy(gfootball_il_main_config), deepcopy(gfootball_il_create_config)]
collect_demo_data(
    collect_config, seed=seed, expert_data_path=data_path_transitions, collect_count=demo_transitions,
    model=football_kaggle_5th_place_model, state_dict=state_dict,
)

"""
phase 2: il training
"""
il_config = [deepcopy(gfootball_il_main_config), deepcopy(gfootball_il_create_config)]
il_config[0].policy.learn.train_epoch = 1000  # key hyper-parameter
il_config[0].env.stop_value = 999  # Don't stop until training <train_epoch> epochs
il_config[0].policy.eval.evaluator.multi_gpu = False
football_naive_q = FootballNaiveQ()
_, converge_stop_flag = serial_pipeline_bc(il_config, seed=seed, data_path=data_path_transitions,
                                           model=football_naive_q)
