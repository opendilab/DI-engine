# DI-engine Gfootball Readme


## Structure

``dizoo/gfootball``目录下的文件大致结构如下：

```
├── README.md
├── __init__.py
├── entry
│   ├── __init__.py
│   ├── gfootball_il_default_config.py
│   └── parallel
├── envs
│   ├── __init__.py
│   ├── action
│   ├── fake_dataset.py
│   ├── gfootball_env.py
│   ├── gfootballsp_env.py
│   ├── obs
│   ├── reward
│   └── tests
├── gfootball.gif
├── model
│   ├── __init__.py
│   ├── bots
│   ├── conv1d
│   └── iql
├── policy
│   ├── __init__.py
│   └── ppo_lstm.py
└── replay.py
```



其中：

- entry：存放gfootball环境对应算法的config配置

- envs：存放DI-engine gfootball环境对应代码

- model：存放gfootball环境相关模型：

  - iql：用于进行independent Q learning的神经网络模型

  - conv1d：用于进行ppo self play training的神经网络模型

  - bots：gfootball对应的一些已有的AI转换为神经网络模型



## Environment

Gfootball环境即Google Research Football环境，其开源代码和对应安装方式可见 https://github.com/google-research/football

DI-engine对Google Research Football 环境进行了封装，使之符合DI-engine环境对应接口，方便使用。具体使用方式可以见 ``dizoo/gfootball/envs/tests/test_env_gfootball.py`` 

目前DI-engine的Gfootball环境支持与内置AI进行对战，后续会设计接口支持双方对战。

目前DI-engine的Gfootball环境支持replay保存，环境config设置 ``save_replay=True`` 后会自动保存replay，包括一个.avi视频文件和一个.dump文件，保存在当前文件夹内的 ``./tmp/football`` 文件夹下。.avi形式的视频默认为2d表示。



如果需要立体表示（真实游戏画面），可以找到对应episode的 .dump文件，然后使用replay.py渲染视频，示例如下：

```python
python replay.py --trace_file=\tmp\football\episode_done_20210331-132800614938.dump
```





## Model

Model分为模型部分和bot部分。

### iql

iql下存放DI-engine进行independent Q learning的模型。

### bots

bots目前包括

*注：所有bot均来源于Google Research Football with Manchester City F.C. 的kaggle比赛社区。*

- 编写的基于规则的rule_based_bot。Hard code机器人来源于kaggle比赛的社区，这一机器人为社区RL bot提供了众多用于模仿学习的素材。在DI-engine中此bot的代码修改自 https://www.kaggle.com/eugenkeil/simple-baseline-bot。
  
- Kaggle比赛第五名的RL模型，在DI-engine的中用于为IQL提供模仿学习素材。我们的代码修改自 https://github.com/YuriCat/TamakEriFever ，ikki407 & yuricat关于这份优秀工作的分享详见 https://www.kaggle.com/c/google-football/discussion/203412 。

### conv1d

对同队队友采用conv1d进行特征提取的模型，并使用LSTM。在此模型上使用selfplay进行训练100k episode可以得到对战built-in hard AI 80%以上的胜率。提供模型的连接如下：https://drive.google.com/file/d/1O1I3Mcjnh9mwAVDyqhp5coksTDPqMZmG/view?usp=sharing

我们同时提供了使用此模型训练得到的足球AI与游戏内置的AI对战一局的录像，左侧队伍由我们训练得到的模型控制，以4-0战胜了内置AI。该录像的连接如下：
https://drive.google.com/file/d/1n-_bF63IQ49b-p0nEZt_NPTL-dmNkoKs/view?usp=sharing

## Uses

### Imitation learning

目前编写了模仿学习相关入口，直接以iql模型，以Kaggle比赛第五名的RL模型为标签进行监督学习，具体请见`dizoo/gfootball/entry`下相关config入口文件。

### Self Play PPO 

编写了使用self-play的PPO算法进行训练的入口，使用DI-engine提供的league模块和PPO算法。具体请见`dizoo/gfootball/entry/parallel/gfootball_ppo_parallel_config.py`入口。

