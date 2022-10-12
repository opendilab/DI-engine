# DI-engine Gfootball


## Structure

``dizoo/gfootball``目录的文件结构大致如下：

```
├── README.md
├── __init__.py
├── config
│   ├── gfootball_counter_mappo_config.py
│   ├── gfootball_counter_masac_config.py
│   ├── gfootball_keeper_mappo_config.py
│   └── gfootball_keeper_masac_config.py
├── entry
│   ├── __init__.py
│   ├── gfootball_bc_config.py
│   ├── gfootball_bc_kaggle5th_main.py
│   ├── gfootball_bc_rule_lt0_main.py
│   ├── gfootball_bc_rule_main.py
│   ├── gfootball_dqn_config.py
│   └── parallel
│   ├── show_dataset.py
│   ├── test_accuracy.py
├── envs
│   ├── __init__.py
│   ├── action
│   ├── fake_dataset.py
│   ├── gfootball_academy_env.py
│   ├── gfootball_env.py
│   ├── gfootballsp_env.py
│   ├── obs
│   ├── reward
│   └── tests
├── gfootball.gif
├── model
│   ├── __init__.py
│   ├── bots
│   ├── conv1d
│   └── q_network
├── policy
│   ├── __init__.py
│   └── ppo_lstm.py
└── replay.py
```


其中：

- config: 存放``gfootball_academy_env``环境对应的多智能体算法配置

- entry：存放``gfootball_env``环境对应的模仿学习和强化学习算法配置和相关工具函数

- envs：存放gfootball环境: ``gfootball_academy_env``, ``gfootball_env``, ``gfootballsp_env`` 以及 ``obs``, ``action``, ``reward``处理函数

- model：存放gfootball模型：

  - q_network：用于进行模仿学习和强化学习的神经网络模型及其默认设置

  - conv1d：用于进行``ppo self play training``的神经网络模型

  - bots：gfootball环境上已有的基于规则或学习好的专家模型



## Environment

Gfootball 环境即 Google Research Football 环境，其开源代码和安装方式参见: https://github.com/google-research/football.

DI-engine 对 Google Research Football 环境进行了封装，使之符合 DI-engine 环境对应接口，方便使用。具体使用方式参考 ``dizoo/gfootball/envs/tests/test_env_gfootball.py`` 

目前 DI-engine 的 Gfootball 环境支持与内置 AI 进行对战，后续会设计接口支持双方对战。

目前 DI-engine 的 Gfootball 环境支持保存 replay，环境 config 中设置 ``save_replay=True`` 后会自动保存 replay，包括一个.avi视频文件和一个.dump文件，保存在当前文件目录的 ``./tmp/football`` 文件夹下。.avi形式的视频默认为2d表示。



如果需要立体表示（真实游戏画面），可以找到对应 episode 的 .dump文件，然后使用 ``replay.py`` 渲染视频，示例如下：

```python
python replay.py --trace_file=\tmp\football\episode_done_20210331-132800614938.dump
```



## Model

Model分为bot部分和模型部分。

### bots

bots目前包括:

*注：所有bot均来源于Google Research Football with Manchester City F.C. 的kaggle比赛社区。*

- 基于规则的`rule_based_bot_model`。Hard code 机器人来源于 kaggle 比赛的社区，这一机器人为社区RL bot提供了众多用于模仿学习的素材。在DI-engine中此bot的代码修改自 https://www.kaggle.com/eugenkeil/simple-baseline-bot。
  
- Kaggle比赛第五名的RL模型 ``kaggle_5th_place_model.py``，在 DI-engine 中用于提供模仿学习素材。我们的代码修改自 https://github.com/YuriCat/TamakEriFever ，ikki407 & yuricat关于这份优秀工作的介绍详见 https://www.kaggle.com/c/google-football/discussion/203412 。

### q_network

``q_network``路径下存放模仿学习和强化学习的模型及其默认设置。

### conv1d

对同队队友采用 ``conv1d`` 进行特征提取的模型，并使用 LSTM。在此模型上使用 selfplay 训练100k episode后对战 built-in hard AI 可以得到80%以上的胜率。最终训练得到的模型参见：https://drive.google.com/file/d/1O1I3Mcjnh9mwAVDyqhp5coksTDPqMZmG/view?usp=sharing

我们同时提供了使用此模型训练得到的足球AI与游戏内置的AI对战一局的录像，左侧队伍是由我们训练得到的模型控制，以4-0战胜了内置AI (右侧队伍)。该录像的连接如下：
https://drive.google.com/file/d/1n-_bF63IQ49b-p0nEZt_NPTL-dmNkoKs/view?usp=sharing

## 入口文件

### Imitation Leaning (Behaviour Cloning)

目前编写了模仿学习相关入口，以``q_network``路径下的``FootballNaiveQ``作为Q网络/策略网络，以基于规则的模型``rule_based_bot_model`` 和 Kaggle比赛第五名的RL模型 ``kaggle_5th_place_model.py`` 为标签进行监督学习，具体请见`dizoo/gfootball/entry`下相关入口文件:

- `gfootball_bc_rule_main.py`
- `gfootball_bc_rule_lt0_main.py`
- `gfootball_bc_kaggle5th_main.py`

### Reinforcement learning

目前使用DQN算法，具体请参见`dizoo/gfootball/entry`下相关入口文件:
- `gfootball_dqn_config.py`

### Self Play PPO (work in progress)

使用self-play的PPO算法进行训练的入口，使用DI-engine提供的league模块和PPO算法。具体请见`dizoo/gfootball/entry/parallel/gfootball_ppo_parallel_config.py`入口。
