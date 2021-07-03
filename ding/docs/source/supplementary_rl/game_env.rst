强化学习常用的游戏环境/RL Game Environments
============================================

对强化学习中常见的游戏环境做一些介绍，对于在科研论文中常用的小型环境（Atari等）只做简单介绍，主要概述一些中型的游戏环境。

科研论文中常用的基础环境
------------------------

OpenAI gym
~~~~~~~~~~

**简介**

gym是强化学习论文实验中最常见的环境，包含\ **Atari游戏、Classic
Control、Mujuco控制等**\ 。

**接口**

安装：

.. code:: shell

   pip3 install gym

   pip3 install atari-py

   pip3 install mujuco_py

   pip install box2d-py

   #  or

   git clone https://github.com/openai/gym.git

   cd gym

   pip3 install -e '.[all]'

使用：

.. code:: python

   env = gym.make('envname')
   ob, _ = env.reset()
   ob, r, d, _ = env.step(action)

其中，Mujuco环境的使用需要在官网license认证，也可用pybullet作替代。

`OpenAI gym retro <https://github.com/openai/retro>`__
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**简介**

OpenAI在gym的基础上加入了更多的游戏接入，集成为Retro。在Atari之外，还支持任天堂和世嘉的一些游戏。一个经典的案例是OpenAI在2018年举办的\ `Sonic游戏迁移学习比赛 <https://openai.com/blog/retro-contest/>`__\ ，便是建立在Retro环境上。支持的游戏列表如下：

-  Atari

   -  Atari2600 (via Stella)

-  NEC

   -  TurboGrafx-16/PC Engine (via Mednafen/Beetle PCE Fast)

-  Nintendo

   -  Game Boy/Game Boy Color (via gambatte)

   -  Game Boy Advance (via mGBA)

   -  Nintendo Entertainment System (via FCEUmm)

   -  Super Nintendo Entertainment System (via Snes9x)

-  Sega

   -  GameGear (via Genesis Plus GX)

   -  Genesis/Mega Drive (via Genesis Plus GX)

   -  Master System (via Genesis Plus GX)

由于涉及到游戏版权问题，retro只提供了无需商业授权的ROM用于游戏测试和智能体训练。列表如下：

-  `the 128 sine-dot <http://www.pouet.net/prod.php?which=2762>`__ by
   Anthrox

-  `Sega Tween <https://pdroms.de/files/gamegear/sega-tween>`__ by Ben
   Ryves

-  `Happy 10! <http://www.pouet.net/prod.php?which=52716>`__ by Blind IO

-  `512-Colour Test
   Demo <https://pdroms.de/files/pcengine/512-colour-test-demo>`__ by
   Chris Covell

-  `Dekadrive <http://www.pouet.net/prod.php?which=67142>`__ by
   Dekadence

-  `Automaton <https://pdroms.de/files/atari2600/automaton-minigame-compo-2003>`__
   by Derek Ledbetter

-  `Fire <http://privat.bahnhof.se/wb800787/gb/demo/64/>`__ by dox

-  `FamiCON intro <http://www.pouet.net/prod.php?which=53497>`__ by dr88

-  `Airstriker <https://pdroms.de/genesis/airstriker-v1-50-genesis-game>`__
   by Electrokinesis

-  `Lost
   Marbles <https://pdroms.de/files/gameboyadvance/lost-marbles>`__ by
   Vantage

其余游戏均需要自行下载ROM使用，可在\ `Archive.org <https://archive.org/details/No-Intro-Collection_2016-01-03_Fixed>`__\ 下载，导入方式如下：

.. code:: python

   python3 -m retro.import /path/to/your/ROMs/directory/

**接口**

安装：

.. code:: 

   pip3 install gym-retro

使用：

与openAI gym的接口保持一致。

`Gridworld <https://github.com/Bigpig4396/Multi-Agent-Reinforcement-Learning-Environment>`__\ 、\ `MiniGrid <https://github.com/maximecb/gym-minigrid>`__
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**简介**

GridWorld和MiniGrid两个都是强化学习中讨论探索利用问题和多智能体问题常用的环境，即二维走迷宫探索环境，实现简单，且容易修改定制地图本身和目标任务。其中，Gridworld的部分环境支持多智能体环境，MiniGrid只有单智能体相关的环境。

.. image:: images/GridWorld.png
   :alt: 

.. image:: images/MiniGrid.png
   :alt: 

**接口**

安装：

GridWorld未提交pip包管理服务器，需要git到本地目录导入。

MiniGrid直接通过\ ``pip3 install gym-minigrid``\ 安装。

使用：

与openAI gym保持一致。

`Multiagent Particle <https://github.com/openai/multiagent-particle-envs>`__
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**简介**

在nerveX中已有实现。Particle为OpenAI为研究多智能体之间的合作、竞争、通讯开发的强化学习环境，智能体的数量和目标任务都可以自定义设置，可以创建超大量级的协作粒子数，本身为MADDPG论文使用的环境。与之类似的还有UCL汪军团队开发的\ `MAgent <https://github.com/geek-ai/MAgent>`__\ 环境。


**接口**

安装：

均需要到连接中git源码，对于Particle：

.. code:: shell

   pip install -e .

对于MAgent：

.. code:: shell

   bash build.sh

使用：

与openAI gym保持一致。

`ProcGen <https://openai.com/blog/procgen-benchmark/>`__
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**简介**

在nerveX中已有实现。ProcGen是OpenAI开发的用于验证强化学习模型迁移和泛化能力的环境。包含16个不同类型的小游戏，每款游戏都有相似类型的不同地图，用于验证模型的知识迁移能力。（官方有PPO算法下200M的训练，有收敛保证）

.. image:: images/ProcGen.png
   :alt: 

**接口**

安装：

.. code:: shell

   pip3 install procgen

使用：

与openAI gym保持一致。

已有训练相关python接口的中型游戏环境
------------------------------------

+----------+----------+----------+----------+----------+----------+
| 环境名称 | 游戏类型 | 状态空间 | 动作空间 | 奖励     | 备注     |
+==========+==========+==========+==========+==========+==========+
| Malmo    | 沙盒     | 图像     | 离散的   | 稀疏，   |          |
|          |          |          | 键盘映射 | 在挖到钻 |          |
|          |          |          |          | 石时获得 |          |
+----------+----------+----------+----------+----------+----------+
| Obstacle | 解谜     | 图像     | 离散的   | 有游戏内 | Exp      |
| Tower    |          |          | 键盘映射 | 的dense  | loration |
|          |          |          |          | 评价分数 | &        |
|          |          |          |          |          | Expl     |
|          |          |          |          |          | oitation |
+----------+----------+----------+----------+----------+----------+
| Torcs    | 赛车     | 图       | 离散的   | 通常     | Transfer |
|          |          | 像或连续 | 键盘映射 | 根据任务 | Learning |
|          |          | 的车路信 |          | 自行设计 |          |
|          |          | 息vector |          |          |          |
+----------+----------+----------+----------+----------+----------+
| DeepMind |          | 图像     | 离散的   |          |          |
| Lab      |          |          | 键盘映射 |          |          |
+----------+----------+----------+----------+----------+----------+
| VizDoom  | FPS      | 图像和状 | 离散的   | 通常会   | Sparse   |
|          |          | 态vector | 键盘映射 | 自行设计 | Reward,  |
|          |          |          |          | （拾取、 | Exp      |
|          |          |          |          | 击败等） | loration |
|          |          |          |          |          | &        |
|          |          |          |          |          | Expl     |
|          |          |          |          |          | oitation |
+----------+----------+----------+----------+----------+----------+
| P        | 休闲     | 地       | 离散的   | 稀疏奖   | POMDP,   |
| ommerman |          | 图特征ve | 键盘映射 | 励，在击 | Sparse   |
|          |          | ctor及状 |          | 败时获得 | Reward,  |
|          |          | 态vector |          |          | Exp      |
|          |          |          |          |          | loration |
|          |          |          |          |          | &        |
|          |          |          |          |          | Expl     |
|          |          |          |          |          | oitation |
+----------+----------+----------+----------+----------+----------+
| Quake    | FPS      | 图像     | 离散的   | 稀疏奖励 | Mul      |
| III      |          |          | 键盘映射 | ，在预定 | tiAgent, |
|          |          |          |          | 时间拥有 | Sparse   |
|          |          |          |          | Flag获得 | Reward,  |
|          |          |          |          |          | Exp      |
|          |          |          |          |          | loration |
|          |          |          |          |          | &        |
|          |          |          |          |          | Expl     |
|          |          |          |          |          | oitation |
+----------+----------+----------+----------+----------+----------+
| Google   | 体育     | 图像或   | 离散的   | 稀疏     | Mul      |
| Research |          | 连续的状 | 键盘映射 | 奖励，进 | tiAgent, |
| Football |          | 态vector |          | 球时获得 | Sparse   |
|          |          |          |          |          | Reward   |
+----------+----------+----------+----------+----------+----------+
| Neural   | MMORPG   | 图像     | 离散的   | 生存时   | Exp      |
| MMOs     |          |          | 键盘映射 | 间，通常 | loration |
|          |          |          |          | 根据任务 | &        |
|          |          |          |          | 自行设计 | Expl     |
|          |          |          |          |          | oitation |
+----------+----------+----------+----------+----------+----------+
| Fever    | 体育     | ve       | 离散的   | 稀疏     | Sparse   |
| Ba       |          | ctor信息 | 键盘映射 | 奖励，得 | Reward,  |
| sketball |          |          |          | 分时获得 | Exp      |
|          |          |          |          |          | loration |
|          |          |          |          |          | &        |
|          |          |          |          |          | Expl     |
|          |          |          |          |          | oitation |
+----------+----------+----------+----------+----------+----------+
| SMAC     | RTS      | ve       | 离散的   | 系数奖励 | Sparse   |
|          |          | ctor信息 | 键盘映射 | ，胜利方 | Reward,  |
|          |          |          |          | +1，负方 | Multi    |
|          |          |          |          | -1。也内 | Agent    |
|          |          |          |          | 置了带有 |          |
|          |          |          |          | 击杀奖励 |          |
|          |          |          |          | 的设置。 |          |
+----------+----------+----------+----------+----------+----------+

`Malmo <https://github.com/Microsoft/malmo>`__
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**简介**

Malmo是微软基于Minecraft开发的AI研究环境，本质上还是一个开放世界的环境，本身不涉及特定的任务。但可以在其上建立相对应的环境来实现任务设计，例如微软在17年在Malmo环境上做过合作抓猪的比赛，20年做了挖矿比赛。环境本身有和Java的Minecraft客户端直接通讯实现的版本，和python
based的版本。与java通讯的版本可以使用较大量的原生Minecraft实例，但与gym
API的不匹配情况也比较严重，纯python的版本可用的实例较少，但对于强化学习算法兼容性更好，且不需要编译java端的代码。

状态空间：图像RGB

动作空间：离散，对应键盘映射

**接口**

安装：

在win10，Linux和MacOS均可以安装。按照\ `此链接 <https://github.com/Microsoft/malmo/blob/master/scripts/python-wheel/README.md>`__\ 在各个平台上安装。几个重要的依赖项：

-  Java8 JDK（需将JAVA_HOME加入环境变量）

-  git

-  ffmpeg

也可以通过docker直接构建

`Obstacle Tower <https://github.com/Unity-Technologies/obstacle-tower-env>`__
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**简介**

基于Unity实现的一个类似三维魔塔的爬楼+解谜游戏。在AAAI
2020上被推出，并附有gym
interface。控制的状态空间为图像，动作空间为离散，包括WSAD方向，KL左右转视角和Space跳跃七维。（官方有使用Rainbow的训练实现）

.. image:: images/ObstacleTower.png
   :alt: 

状态空间：图像

动作空间：离散，对应键盘映射

**接口**

安装：

-  下载\ `游戏渲染程序 <https://github.com/Unity-Technologies/obstacle-tower-env#download-the-environment-optional>`__\ ；

-  git python工程源码并安装依赖项；

.. code:: shell

   git clone git@github.com:Unity-Technologies/obstacle-tower-env.git
   cd obstacle-tower-env
   pip install -e .

-  将游戏程序的ObstacleTower文件夹复制到python工程目录下即可。

**使用**

.. code:: python

   from obstacle_tower_env import ObstacleTowerEnv, ObstacleTowerEvaluation
   env = ObstacleTowerEnv("./ObstacleTower/obstacletower")
   env = ObstacleTowerEvaluation(env, seeds)

其余部分使用方式与openAI gym保持一致。

`Torcs <https://link.zhihu.com/?target=https%3A//github.com/ugo-nama-kun/gym_torcs>`__
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Torcs是一个RL领域比较出名的赛车环境。环境的输入为与现实情况比较接近的路侧距离等传感器信息或者图像信息，车辆本身的各项指标也都可定义，也提供了不同的地图供训练尝试。（官方有DDPG实现）

.. image:: images/Torcs.png
   :alt: 

状态空间：连续的车路信息vector或图像

动作空间：离散，对应键盘映射

**接口**

安装：

仅在ubuntu环境下适用，需要安装依赖：

-  `xautomation <http://linux.die.net/man/7/xautomation>`__

-  gym

-  `vtorcs-RL-color <https://github.com/giuse/vtorcs/tree/nosegfault>`__

特别的，如果不需要处理RGB，在ubuntu上只需要：

.. code:: shell

   sudo apt-get install xautomation

然后安装：

.. code:: shell

   pip3 install gym_torcs

需要渲染时，在不同平台需要安装对应的torcs软件。

使用：

.. code:: python

   from gym_torcs import TorcsEnv
   env = TorcsEnv(vision=True, throttle=False)
   ob = env.reset(relaunch=True)  # with torcs relaunch (avoid memory leak bug in torcs)
   from sample_agent import Agent
   agent = Agent(1)  # steering only
   action = agent.act(ob, reward, done, vision=True)
   ob, reward, done, _ = env.step(action)
   env.end()

基本与OpenAI gym保持一致。

.. _deepmind-lab--hard-eight:

`DeepMind Lab <https://github.com/deepmind/lab>`__
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**简介**

DeepMind Lab是DeepMind在IMPALA论文中使用的环境，为3D导航探索任务。

.. image:: images/DeepMindLab.png
   :alt: 



在官方github上都提供了简单的python接口安装方式。

`VizDoom <https://github.com/mwydmuch/ViZDoom>`__
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**简介**

VizDoom是一个经典的FPS游戏，也是在RL里做过比赛的游戏环境。游戏本身可以使用不同武器（从地图中收集获取），目标是生存并击败对手。仿真速度很快（7000FPS，通常的游戏节奏~30FPS），对Win、Ubuntu和MacOS都可以支持，并支持自定义场景。

.. image:: images/VizDoom.png
   :alt: 

官方在16-18年举行了三届比赛，每次都是单人+多人死亡竞赛两条赛道。三年排名靠前的参赛者都是同一批人（Arnold、TSAIL和IntelAct），但游戏实际表现都未到达人类玩家的水平。TSAIL团队提供了其实现的一些细节，例如使用YOLO-v3作为检测框架来提取特征信息，并使用了分层强化学习的思路来训练agent。在\ `AAAI2017的论文中 <https://ojs.aaai.org/index.php/AAAI/article/view/10827>`__\ ，也提到了在训练中采用目标检测框架来增加feature帮助RL算法的细节，其RL算法使用了DRQN。

状态空间：图像+状态vector。前者通常为30*45的图像，后者包含一些弹药情况、武器情况信息。

动作空间：离散，对应键盘映射。但也可以包含对应鼠标控制的连续量，通常将之离散化来操作。

**接口**

安装：

.. code:: shell

   sudo apt install cmake libboost-all-dev libsdl2-dev libfreetype6-dev libgl1-mesa-dev libglu1-mesa-dev libpng-dev libjpeg-dev libbz2-dev libfluidsynth-dev libgme-dev libopenal-dev zlib1g-dev timidity tar nasm
   pip install vizdoom

使用：

与openAI gym形式上相近，但细节稍有不同：

.. code:: python

   from vizdoom import *
   import random
   import time

   game = DoomGame()
   game.load_config("vizdoom/scenarios/basic.cfg")
   game.init()

   shoot = [0, 0, 1]
   left = [1, 0, 0]
   right = [0, 1, 0]
   actions = [shoot, left, right]

   episodes = 10
   for i in range(episodes):
       game.new_episode()
       while not game.is_episode_finished():
           state = game.get_state()
           img = state.screen_buffer
           misc = state.game_variables
           reward = game.make_action(random.choice(actions))
           print "\treward:", reward
           time.sleep(0.02)
       print "Result:", game.get_total_reward()
       time.sleep(2)

由于举办过VizDoom的比赛，因此相关的实例和一些算法的参考实现相对充足，可以参见\ `tutorial <http://vizdoom.cs.put.edu.pl/tutorial>`__\ 。

`Pommerman <https://www.pommerman.com/>`__
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**简介**

经典炸弹人小游戏，也是Nips2018竞赛的环境。涉及到了强化学习可能面对的探索利用、部分可观、多智能体和资源利用等多方面的问题。通常的版本为四个智能体，可以分别指定各个智能体使用的策略已进行自搏等训练。除了官方实现外，还有很多可以参考的a2c、ppo实现。

.. image:: images/Pommerman.png
   :alt: 

状态空间：地图特征vector及状态vector

-  **Board:** 121 Ints。agent 无法观测到的单位被标记为 5（迷雾）。

-  **Position:** 2 Ints，大小 [0, 10]。agent 在游戏 Board 上的 (x, y)
   位置坐标。

-  **Ammo:** 1 Int。agent 当前可以使用的炸弹数量。

-  **Blast Strength:** 1 Int.。agent 施放炸弹的爆炸范围。

-  **Can Kick:** 1 Int，布尔变量。是否 agent 能踢炸弹。

-  **Teammate:** 1 Int，范围 [-1, 3]. 当前 agent 的队友为哪个。

-  **Enemies:** 3 Ints，范围 [-1, 3]。当前 agent 的敌人是哪些。如果是
   2v2，那么第三个数值为 - 1。

-  **Bombs:** List of Ints。agent 视野范围内的炸弹，通过三元数组表示（x
   int, y int, blast_strength int），表示炸弹位置
   x、y，以及炸弹爆炸范围。

动作空间：离散，对应键盘映射

-  **Stop:** 静止不动

-  **Up:** 向上走

-  **Left:** 向左走

-  **Down:** 向下走

-  **Right:** 向右走

-  **Bomb:** 放置一个炸弹

**接口**

安装：

.. code:: shell

   git clone https://github.com/MultiAgentLearning/playground ~/playground
   cd ~/playground
   pip install -U .

使用：

.. code:: python

   import pommerman
   from pommerman import agents
   def main():
       agent_list = [
           agents.SimpleAgent(),
           agents.RandomAgent(),
           agents.SimpleAgent(),
           agents.RandomAgent(),
           # agents.DockerAgent("pommerman/simple-agent", port=12345),
       ]
       # Make the "Free-For-All" environment using the agent list
       env = pommerman.make('PommeFFACompetition-v0', agent_list)
       # Run the episodes just like OpenAI Gym
       for i_episode in range(1):
           state = env.reset()
           done = False
           while not done:
               env.render()
               actions = env.act(state)
               state, reward, done, info = env.step(actions)
           print('Episode {} finished'.format(i_episode))
       env.close()

与openAI gym类似，但由于是多智能体环境，需要指定每个智能体的策略。

`Quake III Arena Capture the Flag <https://github.com/deepmind/lab>`__
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**简介**

经典游戏雷神之锤夺旗竞技场地图，本身也是集成在DeepMind
Lab下的环境。游戏分为两队，每队有两个智能体，在不同的地图中以第一人称视角进行夺旗游戏。因为DeepMind在这个环境上做训练的成果发在了\ `Science <https://deepmind.com/blog/article/capture-the-flag-science>`__\ 上，因此比较出名。DeepMind在这里用了population
based的训练方法，在延迟0.26秒的反应时间前提下获得了超越人类玩家的智能体。训练框架仅在linux下可用。

.. image:: images/QuakeCTF.png
   :alt: 

状态空间：图像，大小可自定义

动作空间：本身为连续动作空间，但通常会进行离散化到键盘映射。固定为几个确定的动作模式。例如：

.. code:: python

     ACTIONS = {
         'look_left': _action(-20, 0, 0, 0, 0, 0, 0),
         'look_right': _action(20, 0, 0, 0, 0, 0, 0),
         'look_up': _action(0, 10, 0, 0, 0, 0, 0),
         'look_down': _action(0, -10, 0, 0, 0, 0, 0),
         'strafe_left': _action(0, 0, -1, 0, 0, 0, 0),
         'strafe_right': _action(0, 0, 1, 0, 0, 0, 0),
         'forward': _action(0, 0, 0, 1, 0, 0, 0),
         'backward': _action(0, 0, 0, -1, 0, 0, 0),
         'fire': _action(0, 0, 0, 0, 1, 0, 0),
         'jump': _action(0, 0, 0, 0, 0, 1, 0),
         'crouch': _action(0, 0, 0, 0, 0, 0, 1)
     }

**接口**

安装：

-  安装\ `bazel <https://docs.bazel.build/versions/master/install.html>`__

-  git
   Deepmind提供的python框架源码\ ``git clone https://github.com/deepmind/lab``

-  在\ ``/lab/python/pip_package``\ 中\ ``pip install -e .``\ 安装相关依赖包

使用：

提供了直接作为玩家接入的模式和智能体训练模式。对于后者，可以参考\ `官方实例 <https://github.com/deepmind/lab/blob/master/python/random_agent.py>`__\ 。用法与上面的pommerman接近，需选定agent和env类型。

`Google Research Football <https://github.com/google-research/football>`__
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**简介**

这个环境是 google
基于之前某个足球小游戏的环境进行改动和封装出来的，主要可以分为 11v11
single-agent 场景（控制一个 active player 在 11 名球员中切换）和 5v5
multi-agent 场景（控制 4 名球员 + 1 个守门员）。该环境支持
self-play，有三种难度内置 AI 可以打。游戏状态基于 vector
的主要是球员的坐标 / 速度 / 角色 / 朝向 /
红黄牌等，也可以用图像输入，动作输出有二十多维，包括不同方向 / 长短传 /
加速等。是Google在Kaggle上举办过比赛的环境，实际会面对RL中的多智能体、稀疏奖励等多种问题。环境训练本身支持Linux和MacOS。

.. image:: images/GFootball.png
   :alt: 

状态空间：图像或vector信息

动作空间：离散，对应键盘映射

**接口**

安装：

.. code:: shell

   sudo apt-get install git cmake build-essential libgl1-mesa-dev libsdl2-dev \
   libsdl2-image-dev libsdl2-ttf-dev libsdl2-gfx-dev libboost-all-dev \
   libdirectfb-dev libst-dev mesa-utils xvfb x11vnc libsdl-sge-dev python3-pip
   python3 -m pip install --upgrade pip setuptools psutil
   python3 -m pip install gfootball

使用：

官方有内建的tensorflow实例，并使用openAI
baseline来训练。因此整个交互框架与openAI gym相同。

.. code:: python

   import gfootball.env as football_env
   env = football_env.create_environment(env_name="academy_empty_goal_close", stacked=False, logdir='/tmp/football', write_goal_dumps=False, write_full_episode_dumps=False, render=False)
   env.reset()
   steps = 0
   while True:
     obs, rew, done, info = env.step(env.action_space.sample())
     steps += 1
     if steps % 100 == 0:
       print("Step %d Reward: %f" % (steps, rew))
     if done:
       break
   print("Steps: %d Reward: %.2f" % (steps, rew))

`Neural MMOs <https://github.com/openai/neural-mmo>`__
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**简介**

OpenAI开源的大型MultiAgent环境，在非常大的地图中设定有限资源。也因为地图非常大，对IO开销非常大。

.. image:: images/NeuralMMOs.png
   :alt: 

**接口**

安装：

.. code:: shell

   git clone https://github.com/jsuarez5341/neural-mmo-client
   cd neural-mmo-client
   bash setup.sh
   cd ..

   git clone https://github.com/openai/neural-mmo
   cd neural-mmo
   bash scripts/setup/setup.s

使用：

.. code:: shell

   python Forge.py --render #Run the environment with rendering on

.. code:: python

   from forge.trinity import smith
   envs = smith.VecEnv(config, args, self.step)

   #The environment is persistent: call reset only upon initialization
   obs = envs.reset()

   #Observations contain entity and stimulus
   #for each agent in each environment.
   actions = your_algorithm_here(obs)

   #The environment is persistent: "dones" is always None
   #If an observation is missing, that agent has died
   obs, rewards, dones, infos = envs.step(actions)

`Fever Basketball <https://github.com/FuxiRL/FeverBasketball>`__
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**简介**

网易伏羲开源的潮人篮球游戏，支持1v1，2v2，3v3环境，提供内置不同难度的AI，支持self-play。

.. image:: images/FeverBasketball.png
   :alt: 

状态空间：vector信息

动作空间：离散，对应键盘映射

**接口**

安装：

-  安装python工程文件。

.. code:: shell

   git clone https://github.com/FuxiRL/FeverBasketball.git
   pip3 install -r requirements.txt

-  下载\ `游戏客户端 <https://pan.baidu.com/share/init?surl=visZLh5QEXqQakdVOlPqhg>`__

使用：

环境并未用gym的形式进行封装，而是以socket通信的方式与windows客户端程序进行交互来实现step和observe。网易伏羲官方提供了几种RL算法包括PPO、QMIX等的实现（未调）。

`SMAC <https://github.com/canyon/smac>`__
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

NerveX中已有实现。SMAC是调用星际争霸2接口实现的多智能体RL环境，游戏类型为RTS，基于星际争霸2的API和DeepMind的PySC2实现。在星际争霸2的常规完整游戏中，一个或多个人类彼此竞争或与内置游戏
AI 竞争，以收集资源，建造建筑物并建立部队单位以击败对手。SMAC
由一套完整星际争霸2的一小部分组成，旨在评估Agent学习协调解决复杂任务的能力。这些场景经过精心设计，必须学习一种或多种微操技术才能击败敌人。每种情况都是两支部队之间的对抗。每个部队的初始位置，数量和类型随场景的不同而变化。具体包括如下内容：

+----------------+----------------+----------------+----------------+
| Name           | Ally Units     | Enemy Units    | Type           |
+================+================+================+================+
| 3m             | 3 Marines      | 3 Marines      | homogeneous &  |
|                |                |                | symmetric      |
+----------------+----------------+----------------+----------------+
| 8m             | 8 Marines      | 8 Marines      | homogeneous &  |
|                |                |                | symmetric      |
+----------------+----------------+----------------+----------------+
| 25m            | 25 Marines     | 25 Marines     | homogeneous &  |
|                |                |                | symmetric      |
+----------------+----------------+----------------+----------------+
| 2s3z           | 2 Stalkers & 3 | 2 Stalkers & 3 | heterogeneous  |
|                | Zealots        | Zealots        | & symmetric    |
+----------------+----------------+----------------+----------------+
| 3s5z           | 3 Stalkers & 5 | 3 Stalkers & 5 | heterogeneous  |
|                | Zealots        | Zealots        | & symmetric    |
+----------------+----------------+----------------+----------------+
| MMM            | 1 Medivac, 2   | 1 Medivac, 2   | heterogeneous  |
|                | Marauders & 7  | Marauders & 7  | & symmetric    |
|                | Marines        | Marines        |                |
+----------------+----------------+----------------+----------------+
| 5m_vs_6m       | 5 Marines      | 6 Marines      | homogeneous &  |
|                |                |                | asymmetric     |
+----------------+----------------+----------------+----------------+
| 8m_vs_9m       | 8 Marines      | 9 Marines      | homogeneous &  |
|                |                |                | asymmetric     |
+----------------+----------------+----------------+----------------+
| 10m_vs_11m     | 10 Marines     | 11 Marines     | homogeneous &  |
|                |                |                | asymmetric     |
+----------------+----------------+----------------+----------------+
| 27m_vs_30m     | 27 Marines     | 30 Marines     | homogeneous &  |
|                |                |                | asymmetric     |
+----------------+----------------+----------------+----------------+
| 3s5z_vs_3s6z   | 3 Stalkers & 5 | 3 Stalkers & 6 | heterogeneous  |
|                | Zealots        | Zealots        | & asymmetric   |
+----------------+----------------+----------------+----------------+
| MMM2           | 1 Medivac, 2   | 1 Medivac, 3   | heterogeneous  |
|                | Marauders & 7  | Marauders & 8  | & asymmetric   |
|                | Marines        | Marines        |                |
+----------------+----------------+----------------+----------------+
| 2m_vs_1z       | 2 Marines      | 1 Zealot       | micro-trick:   |
|                |                |                | alternating    |
|                |                |                | fire           |
+----------------+----------------+----------------+----------------+
| 2s_vs_1sc      | 2 Stalkers     | 1 Spine        | micro-trick:   |
|                |                | Crawler        | alternating    |
|                |                |                | fire           |
+----------------+----------------+----------------+----------------+
| 3s_vs_3z       | 3 Stalkers     | 3 Zealots      | micro-trick:   |
|                |                |                | kiting         |
+----------------+----------------+----------------+----------------+
| 3s_vs_4z       | 3 Stalkers     | 4 Zealots      | micro-trick:   |
|                |                |                | kiting         |
+----------------+----------------+----------------+----------------+
| 3s_vs_5z       | 3 Stalkers     | 5 Zealots      | micro-trick:   |
|                |                |                | kiting         |
+----------------+----------------+----------------+----------------+
| 6h_vs_8z       | 6 Hydralisks   | 8 Zealots      | micro-trick:   |
|                |                |                | focus fire     |
+----------------+----------------+----------------+----------------+
| corridor       | 6 Zealots      | 24 Zerglings   | micro-trick:   |
|                |                |                | wall off       |
+----------------+----------------+----------------+----------------+
| bane_vs_bane   | 20 Zerglings & | 20 Zerglings & | micro-trick:   |
|                | 4 Banelings    | 4 Banelings    | positioning    |
+----------------+----------------+----------------+----------------+
| so_many        | 7 Zealots      | 32 Banelings   | micro-trick:   |
| _banelings     |                |                | positioning    |
+----------------+----------------+----------------+----------------+
| 2c_vs_64zg     | 2 Colossi      | 64 Zerglings   | micro-trick:   |
|                |                |                | positioning    |
+----------------+----------------+----------------+----------------+

状态空间：vector信息，包括每个单位视野范围（9）内其它单位的信息，包括距离、相对位置、血量、单位类型等。

动作空间：离散。包括移动（向四个方向）、攻击（对于医疗单位为治疗）、停止和无操作，攻击（治疗）需选定视野范围内的目标。

**接口**

安装：

-  安装\ `StarCraft
   II <https://github.com/canyon/smac#installing-starcraft-ii>`__\ （Linux，Win
   or MacOS）。

-  安装python工程。

.. code:: python

   pip install git+https://github.com/oxwhirl/smac.git

使用：

沿用PySC2的接口。

.. code:: python

   # for testing
   python -m smac.examples.random_agents
