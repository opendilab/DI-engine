from ding.entry.application_entry import collect_episodic_demo_data, episode_to_transitions, episode_to_mc_transitions
from dizoo.smac.config.smac_MMM2_masac_config import main_config, create_config
from ding.model.template.maqac import MAQAC
import torch

model = MAQAC(agent_obs_shape=204,global_obs_shape=431,action_shape=18,twin_critic=True,actor_head_hidden_size=256,critic_head_hidden_size=256,)
info = torch.load('/home/weiyuhong/MASAC_bc/DI-engine/smac_MMM2_masac_5e5_1600/ckpt/ckpt_best.pth.tar',map_location = 'cpu')
model.load_state_dict(info['model'])

episode_to_mc_transitions(data_path='./MMM2_expert.pkl',expert_data_path='./MMM2_mc_expert.pkl',gamma=0.99,)

#episode_to_transitions(data_path='./MMM2_expert.pkl',expert_data_path='./MMM2_ttt.pkl',nstep=3,)