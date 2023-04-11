from ding.bonus import C51Agent
from huggingface_ding import push_model_to_hub

# Instantiate the agent
agent = C51Agent("lunarlander_discrete", exp_name="LunarLander-v2-C51")
# Train the agent
return_ = agent.train(step=200000)
# Push model to huggingface hub
""" push_model_to_hub(
    agent=agent.best,
    env_name="OpenAI/Gym/Box2d",
    task_name="LunarLander-v2",
    algo_name="DDPG",
    wandb_url=return_.wandb_url,
    github_repo_url="https://github.com/opendilab/DI-engine",
    github_doc_model_url="https://di-engine-docs.readthedocs.io/en/latest/12_policies/ddpg.html",
    github_doc_env_url="https://di-engine-docs.readthedocs.io/en/latest/13_envs/lunarlander.html",
    installation_guide="pip3 install DI-engine[common_env]",
    usage_file_by_git_clone="./dizoo/common/ddpg/lunarlander_ddpg_deploy.py",
    usage_file_by_huggingface_ding="./dizoo/common/ddpg/lunarlander_ddpg_download.py",
    train_file="./dizoo/common/ddpg/lunarlander_ddpg.py",
    repo_id="OpenDILabCommunity/LunarLander-v2-DDPG"
) """