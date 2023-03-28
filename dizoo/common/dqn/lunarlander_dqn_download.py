from ding.bonus import DQNOffpolicyAgent
from huggingface_ding import pull_model_from_hub

# Pull model from Hugggingface hub
policy_state_dict, cfg=pull_model_from_hub(repo_id="OpenDILabCommunity/Lunarlander-v2-DQN")
# Instantiate the agent
agent = DQNOffpolicyAgent(env="lunarlander_discrete",exp_name="Lunarlander-v2-DQN-test", cfg=cfg.exp_config, policy_state_dict=policy_state_dict)
# Continue training
agent.train(step=5000)
# Render the new agent performance
agent.deploy(enable_save_replay=True)
