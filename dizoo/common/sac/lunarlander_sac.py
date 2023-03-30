from ding.bonus import SACAgent
from huggingface_ding import push_model_to_hub

# Instantiate the agent
agent = SACAgent("lunarlander_continuous", exp_name="LunarLander-v2-SAC")
# Train the agent
return_ = agent.train(step=int(4000000), collector_env_num=8, evaluator_env_num=8)
# Push model to huggingface hub
push_model_to_hub(
    agent=agent,
    env_name="OpenAI/Gym/Box2d",
    task_name="LunarLander-v2",
    algo_name="SAC",
    wandb_url=return_.wandb_url,
    github_repo_url="https://github.com/opendilab/DI-engine",
    github_doc_model_url="https://di-engine-docs.readthedocs.io/en/latest/12_policies/sac.html",
    github_doc_env_url="https://di-engine-docs.readthedocs.io/en/latest/13_envs/lunarlander.html",
    installation_guide="pip3 install DI-engine[common_env]",
    usage_file_by_git_clone="./dizoo/common/sac/lunarlander_sac_deploy.py",
    usage_file_by_huggingface_ding="./dizoo/common/sac/lunarlander_sac_download.py",
    train_file="./dizoo/common/sac/lunarlander_sac.py",
    repo_id="OpenDILabCommunity/LunarLander-v2-SAC"
)
