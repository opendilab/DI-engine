from ding.bonus import A2CAgent
from huggingface_ding import pull_model_from_hub

# Pull model from Hugggingface hub
policy_state_dict, cfg = pull_model_from_hub(repo_id="OpenDILabCommunity/Lunarlander-v2-A2C")
# Instantiate the agent
agent = A2CAgent(
    env="lunarlander_discrete", exp_name="Lunarlander-v2-A2C", cfg=cfg.exp_config, policy_state_dict=policy_state_dict
)
# Continue training
agent.train(step=5000)
# Render the new agent performance
agent.deploy(enable_save_replay=True)
