# NerveX Gfootball Readme

[TOC]

## Structure

app_zoo/gfootball文件大致结构如下：

├── entry
├── envs
├── model
│   ├── iql
│   └── bots



其中：

- entry：存放gfootball环境对应算法的config配置

- envs：存放nerveX gfootball环境对应代码

- model：存放gfootball环境相关模型：

  - iql：用于进行independent Q learning的神经网络模型

  - bots：gfootball对应的一些已有的AI转换为神经网络模型



## Environment

Gfootball环境即Google Research Football环境，其开源代码和对应安装方式可见 https://github.com/google-research/football

nerevX对Google Research Football 环境进行了封装，使之符合nerevX环境对应接口，方便使用。具体使用方式可以见 app_zoo/gfootball/envs/tests/test_env_gfootball.py 

目前NerveX的Gfootball环境支持与内置AI进行对战，后续会设计接口支持双方对战。



## Model

Model分为模型部分和bot部分。

### iql

iql下存放nerveX进行independent Q learning的模型。

### bots

bots目前包括

- 编写的基于规则的rule_based_bot
- Kaggle比赛第五名的RL模型， 详见https://www.kaggle.com/c/google-football/discussion/203412。





