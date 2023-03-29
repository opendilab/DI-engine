from ding.bonus import SACOffPolicyAgent
from huggingface_ding import pull_model_from_hub

# Pull model from Hugggingface hub
policy_state_dict, cfg = pull_model_from_hub(repo_id="OpenDILabCommunity/LunarLander-v2-SAC")
# Instantiate the agent
agent = SACOffPolicyAgent(
    env="lunarlander_continuous",
    exp_name="LunarLander-v2-SAC",
    cfg=cfg.exp_config,
    policy_state_dict=policy_state_dict
)
# Continue training
agent.train(step=5000)
# Render the new agent performance
agent.deploy(enable_save_replay=True)
