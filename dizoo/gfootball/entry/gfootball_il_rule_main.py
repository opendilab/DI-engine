from copy import deepcopy
import os
import torch

path = os.path.abspath(__file__)
dir_path = os.path.dirname(path)
from ding.entry import serial_pipeline_bc, collect_demo_data
from ding.config import read_config, compile_config
from ding.policy import create_policy
from dizoo.gfootball.entry.gfootball_il_config import gfootball_il_main_config, gfootball_il_create_config
from dizoo.gfootball.model.q_network.football_q_network import FootballNaiveQ
from dizoo.gfootball.model.bots.rule_based_bot_model import FootballRuleBaseModel

# in gfootball env: 3000 transitions = one episode
# 3e5 transitions = 100 episode, The memory needs about 180G
seed = 0
gfootball_il_main_config.exp_name = 'data_gfootball/gfootball_il_rule_seed0_100eps_epc1000_bs512'
demo_transitions = int(3e5)  # key hyper-parameter
data_path_transitions = dir_path + f'/gfootball_rule_{demo_transitions}-demo-transitions.pkl'


"""
phase 1: train/obtain expert policy
"""
input_cfg = [deepcopy(gfootball_il_main_config), deepcopy(gfootball_il_create_config)]
if isinstance(input_cfg, str):
    cfg, create_cfg = read_config(input_cfg)
else:
    cfg, create_cfg = input_cfg
create_cfg.policy.type = create_cfg.policy.type + '_command'
cfg = compile_config(cfg, seed=seed, env=None, auto=True, create_cfg=create_cfg, save_cfg=True)

football_rule_base_model = FootballRuleBaseModel()
expert_policy = create_policy(cfg.policy, model=football_rule_base_model,
                              enable_field=['learn', 'collect', 'eval', 'command'])

# collect expert demo data
state_dict = expert_policy.collect_mode.state_dict()
collect_config = [deepcopy(gfootball_il_main_config), deepcopy(gfootball_il_create_config)]

# eval demo model
# eval_config = deepcopy(collect_config)
# # if save replay
# eval(eval_config, seed=seed, model=football_rule_base_model, replay_path=dir_path + f'/gfootball_rule_replay/')
# # if not save replay
# eval(eval_config, seed=seed, model=football_rule_base_model, state_dict=state_dict)

# collect demo data
collect_demo_data(
    collect_config, seed=seed, expert_data_path=data_path_transitions, collect_count=demo_transitions,
    model=football_rule_base_model, state_dict=state_dict,
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

if il_config[0].policy.test_accuracy:
    """
    phase 3: test accuracy in train dataset and validation dataset
    """
    il_model_path = il_config[0].policy.il_model_path

    # load trained il model
    il_config[0].policy.learn.batch_size = int(3000)
    il_config[0].policy.learn.train_epoch = 1
    il_config[0].policy.learn.show_accuracy = True
    state_dict = torch.load(il_model_path)
    football_naive_q.load_state_dict(state_dict['model'])

    # calculate accuracy in train dataset
    print('==' * 10)
    print('calculate accuracy in train dataset' * 10)
    print('==' * 10)
    train_data_path = dir_path + f'/gfootball_rule_{demo_transitions}-demo-transitions.pkl'
    _, converge_stop_flag = serial_pipeline_bc(il_config, seed=seed, data_path=train_data_path, model=football_naive_q)

    # calculate accuracy in validation dataset
    print('==' * 10)
    print('calculate accuracy in validation dataset' * 10)
    print('==' * 10)
    validation_data_path = dir_path + f'/gfootball_rule_150000-demo-transitions_test.pkl'
    _, converge_stop_flag = serial_pipeline_bc(il_config, seed=seed, data_path=validation_data_path, model=football_naive_q)
