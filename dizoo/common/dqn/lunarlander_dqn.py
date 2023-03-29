from ding.bonus import DQNOffpolicyAgent
from huggingface_ding import push_model_to_hub

# Instantiate the agent
agent = DQNOffpolicyAgent(env="lunarlander_discrete", exp_name="Lunarlander-v2-DQN")
# Train the agent
return_ = agent.train(step=int(4000000), collector_env_num=8, evaluator_env_num=8, debug=False)
# Push model to huggingface hub
push_model_to_hub(
    agent=agent,
    env_name="OpenAI/Gym/Box2d",
    task_name="LunarLander-v2",
    algo_name="DQN",
    wandb_url=return_.wandb_url,
    github_repo_url="https://github.com/opendilab/DI-engine",
    github_doc_model_url="https://di-engine-docs.readthedocs.io/en/latest/12_policies/dqn.html",
    github_doc_env_url="https://di-engine-docs.readthedocs.io/en/latest/13_envs/lunarlander.html",
    installation_guide="pip3 install DI-engine[common_env,fast]",
    usage_file_by_git_clone="./dizoo/common/dqn/lunarlander_dqn_deploy.py",
    usage_file_by_huggingface_ding="./dizoo/common/dqn/lunarlander_dqn_download.py",
    train_file="./dizoo/common/dqn/lunarlander_dqn.py",
    repo_id="OpenDILabCommunity/Lunarlander-v2-DQN"
)
