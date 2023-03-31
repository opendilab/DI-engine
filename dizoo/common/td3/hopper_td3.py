from ding.bonus import TD3Agent
from huggingface_ding import push_model_to_hub

# Instantiate the agent
agent = TD3Agent(env="hopper", exp_name="Hopper-v3-TD3")
# Train the agent
return_ = agent.train(step=int(10000000), collector_env_num=4, evaluator_env_num=4, debug=False)
# Push model to huggingface hub
push_model_to_hub(
    agent=agent,
    env_name="OpenAI/Gym/MuJoCo",
    task_name="Hopper-v3",
    algo_name="TD3",
    wandb_url=return_.wandb_url,
    github_repo_url="https://github.com/opendilab/DI-engine",
    github_doc_model_url="https://di-engine-docs.readthedocs.io/en/latest/12_policies/td3.html",
    github_doc_env_url="https://di-engine-docs.readthedocs.io/en/latest/13_envs/mujoco.html",
    installation_guide=
'''
sudo apt update -y \
    && sudo apt install -y \
    build-essential \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev \
    libglfw3 \
    libglfw3-dev \
    libsdl2-dev \
    libsdl2-image-dev \
    libglm-dev \
    libfreetype6-dev \
    patchelf

mkdir -p ~/.mujoco
wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -O mujoco.tar.gz
tar -xf mujoco.tar.gz -C ~/.mujoco
echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mjpro210/bin:~/.mujoco/mujoco210/bin" >> ~/.bashrc
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mjpro210/bin:~/.mujoco/mujoco210/bin
pip3 install DI-engine[common_env]
''',
    usage_file_by_git_clone="./dizoo/common/td3/hopper_td3_deploy.py",
    usage_file_by_huggingface_ding="./dizoo/common/td3/hopper_td3_download.py",
    train_file="./dizoo/common/td3/hopper_td3.py",
    repo_id="OpenDILabCommunity/Hopper-v3-TD3"
)

