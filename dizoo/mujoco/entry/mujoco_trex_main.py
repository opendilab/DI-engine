import argparse
import torch
from ding.entry import trex_collecting_data
from dizoo.mujoco.config.halfcheetah_trex_onppo_default_config import main_config, create_config
from ding.entry import serial_pipeline_reward_model_trex_onpolicy, serial_pipeline_reward_model_trex

# Note serial_pipeline_reward_model_trex_onpolicy is for on policy ppo whereas serial_pipeline_reward_model_trex is for sac
# Note before run this file, please add the correpsonding path in the config, all path expect exp_name should be abs path

parser = argparse.ArgumentParser()
parser.add_argument(
    '--cfg',
    type=str,
    default='please enter abs path for halfcheetah_trex_onppo_default_config.py or halfcheetah_trex_sac_default_config.py'
)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
args = parser.parse_args()

trex_collecting_data(args)
# if run sac, please import the relevant config and use serial_pipeline_reward_model_trex
serial_pipeline_reward_model_trex_onpolicy([main_config, create_config])
