from ding.bonus import DDPGAgent
from ding.config import Config
from easydict import EasyDict
import torch

# Pull model from files which are git cloned from huggingface
policy_state_dict = torch.load("pytorch_model.bin", map_location=torch.device("cpu"))
cfg = EasyDict(Config.file_to_dict("policy_config.py"))
# Instantiate the agent
agent = DDPGAgent(env="hopper", exp_name="Hopper-v3-DDPG", cfg=cfg.exp_config, policy_state_dict=policy_state_dict)
# Continue training
agent.train(step=5000)
# Render the new agent performance
agent.deploy(enable_save_replay=True)
