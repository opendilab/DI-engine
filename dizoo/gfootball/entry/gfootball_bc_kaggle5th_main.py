"""
Overview:
    Here is the behaviour cloning (BC) main entry for gfootball.
    We first collect demo data using ``FootballKaggle5thPlaceModel``, then train the BC model
    using the collected demo data,
    and (optional) test accuracy in train dataset and test dataset of the trained BC model
"""
from copy import deepcopy
import os
from ding.entry import serial_pipeline_bc, collect_demo_data
from ding.config import read_config, compile_config
from ding.policy import create_policy
from dizoo.gfootball.entry.gfootball_bc_config import gfootball_bc_config, gfootball_bc_create_config
from dizoo.gfootball.model.q_network.football_q_network import FootballNaiveQ
from dizoo.gfootball.model.bots.kaggle_5th_place_model import FootballKaggle5thPlaceModel

path = os.path.abspath(__file__)
dir_path = os.path.dirname(path)

# in gfootball env: 3000 transitions = one episode
# 3e5 transitions = 100 episode, The memory needs about 180G
seed = 0
gfootball_bc_config.exp_name = 'gfootball_bc_kaggle5th_seed0'
demo_transitions = int(3e5)  # key hyper-parameter
data_path_transitions = dir_path + f'/gfootball_kaggle5th_{demo_transitions}-demo-transitions.pkl'
"""
phase 1: collect demo data utilizing ``FootballKaggle5thPlaceModel``
"""
train_config = [deepcopy(gfootball_bc_config), deepcopy(gfootball_bc_create_config)]
input_cfg = train_config
if isinstance(input_cfg, str):
    cfg, create_cfg = read_config(input_cfg)
else:
    cfg, create_cfg = input_cfg
create_cfg.policy.type = create_cfg.policy.type + '_command'
cfg = compile_config(cfg, seed=seed, env=None, auto=True, create_cfg=create_cfg, save_cfg=True)

football_kaggle_5th_place_model = FootballKaggle5thPlaceModel()
expert_policy = create_policy(
    cfg.policy, model=football_kaggle_5th_place_model, enable_field=['learn', 'collect', 'eval', 'command']
)

# collect expert demo data
state_dict = expert_policy.collect_mode.state_dict()
collect_config = [deepcopy(gfootball_bc_config), deepcopy(gfootball_bc_create_config)]
collect_demo_data(
    collect_config,
    seed=seed,
    expert_data_path=data_path_transitions,
    collect_count=demo_transitions,
    model=football_kaggle_5th_place_model,
    state_dict=state_dict,
)
"""
phase 2: BC training
"""
bc_config = [deepcopy(gfootball_bc_config), deepcopy(gfootball_bc_create_config)]
bc_config[0].policy.learn.train_epoch = 1000  # key hyper-parameter
football_naive_q = FootballNaiveQ()
_, converge_stop_flag = serial_pipeline_bc(
    bc_config, seed=seed, data_path=data_path_transitions, model=football_naive_q
)
