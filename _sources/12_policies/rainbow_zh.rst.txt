Rainbow
^^^^^^^

概述
---------
Rainbow 是在 `Rainbow: Combining Improvements in Deep Reinforcement Learning <https://arxiv.org/abs/1710.02298>`_. 它将许多独立的改进方法应用于DQN，包括： Double DQN, priority, dueling head, multi-step TD-loss, C51 (distributional RL) 和 noisy net。

要点摘要
-----------
1. Rainbow 是一种 **无模型（model-free）** 和 **基于值（value-based）** 的强化学习算法。

2. Rainbow 仅支持 **离散动作空间** 。

3. Rainbow 是一种 **异策略（off-policy）** 算法。

4. Usually, Rainbow 使用 **eps-greedy** ， **多项式采样** 或者 **noisy net** 进行探索。

5. Rainbow 可以与循环神经网络 (RNN) 结合使用。

6. Rainbow 的 DI-engine 实现支持 **多离散（multi-discrete）** 动作空间。

关键方程或关键图表
---------------------------

Double DQN
>>>>>>>>>>>>>>>>>
Double DQN, 是在 `Deep Reinforcement Learning with Double Q-learning <https://arxiv.org/abs/1509.06461>`_ 中提出的一种常见的 DQN 变体。传统的DQN维护一个目标Q网络，该网络周期性地使用当前的Q网络进行更新。双重DQN通过解耦解决了Q值的过高估计问题。它使用当前的Q网络选择动作，但使用目标网络估计Q值，具体而言：

.. math::

   \left(R_{t+1}+\gamma_{t+1} q_{\bar{\theta}}\left(S_{t+1}, \underset{a^{\prime}}{\operatorname{argmax}} q_{\theta}\left(S_{t+1}, a^{\prime}\right)\right)-q_{\theta}\left(S_{t}, A_{t}\right)\right)^{2}


Prioritized Experience Replay(PER)
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

DQN 从经验回放缓冲区均匀地进行采样。理想情况下，我们希望更频繁地采样那些有更多可学习内容的 transition。作为评估学习潜力的一种替代方法，优先级经验回放会根据最新的绝对 TD 误差转化得到的概率来采样对应的transition，具体而言：

.. math::

   p_{t} \propto\left|R_{t+1}+\gamma_{t+1} \max _{a^{\prime}} q_{\bar{\theta}}\left(S_{t+1}, a^{\prime}\right)-q_{\theta}\left(S_{t}, A_{t}\right)\right|^{\omega}


在优先级经验回放（PER）的原始论文中，作者展示了在57个Atari游戏中，PER在大多数游戏上都取得了改进，特别是在 Gopher, Atlantis, James Bond 007, Space Invaders 等游戏中。

Dueling Network
>>>>>>>>>>>>>>>
Ddueling network 是一种为基于值的强化学习算法设计的网络架构。它包含两个计算流，一个用于状态值函数 :math:`V` ，另一个用于状态相关的动作优势函数 :math:`A` 。
这两个流共享一个公共的卷积编码器，并通过一个特殊的聚合器合并，产生状态-动作值函数Q的估计，如图所示。

.. image:: images/DuelingDQN.png
           :align: center
           :height: 300
           
给定 :math:`Q` ，我们无法唯一地恢复 :math:`V` 和 :math:`A` 。因此，我们通过以下的动作值分解方式来强制使优势函数为零：

.. math::

   q_{\theta}(s, a)=v_{\eta}\left(f_{\xi}(s)\right)+a_{\psi}\left(f_{\xi}(s), a\right)-\frac{\sum_{a^{\prime}} a_{\psi}\left(f_{\xi}(s), a^{\prime}\right)}{N_{\text {actions }}}

通过这种方式，可以解决可识别性问题，并增加优化过程的稳定性。Rainbow的网络架构是一种针对回报分布进行调整的 dueling network 网络架构。

Multi-step Learning
>>>>>>>>>>>>>>>>>>>
DQN的多步变体通过最小化替代损失来定义，具体如下：


.. math::

   \left(R_{t}^{(n)}+\gamma_{t}^{(n)} \max _{a^{\prime}} q_{\bar{\theta}}\left(S_{t+n}, a^{\prime}\right)-q_{\theta}\left(S_{t}, A_{t}\right)\right)^{2}


其中，截断的n步回报定义为：

.. math::

   R_{t}^{(n)} \equiv \sum_{k=0}^{n-1} \gamma_{t}^{(k)} R_{t+k+1}

在文章 `Revisiting Fundamentals of Experience Replay <https://acsweb.ucsd.edu/~wfedus/pdf/replay.pdf>`_, 作者分析认为，当使用多步学习时，更大容量的回放缓冲区显著提高了性能，并且他们认为原因是多步学习带来了更大的方差，而这一方差可以通过更大的回放缓冲区来进行补偿。

Distributional RL
>>>>>>>>>>>>>>>>>>>
Distributional RL 最初是在 `A Distributional Perspective on Reinforcement Learning <https://arxiv.org/abs/1707.06887>`_ 中提出的。它通过使用离散分布来学习逼近回报的分布，而不是期望回报。它的分布由一个向量 :math:`\boldsymbol{z}` 支持, 即 :math:`z^{i}=v_{\min }+(i-1) \frac{v_{\max }-v_{\min }}{N_{\text {atoms }}-1}` ,其中 :math:`i \in\left\{1, \ldots, N_{\text {atoms }}\right\}`,  :math:`N_{\text {atoms }} \in \mathbb{N}^{+}atoms` 。 
它在t时刻的近似分布 :math:`d_{t}` 在这个支持向量上被定义, 在每个原子 :math:`i` 上的概率为 :math:`p_{\theta}^{i}\left(S_{t}, A_{t}\right)`  最终的分布可以表示为 :math:`d_{t}=\left(z, p_{\theta}\left(S_{t}, A_{t}\right)\right)` 。
然后，通过最小化分布 :math:`d_{t}` 和目标分布之间的Kullback-Leibler散度，得到了一种 Q-learning 的 distributional variant 。

.. math::

   D_{\mathrm{KL}}\left(\Phi_{\boldsymbol{z}} d_{t}^{\prime} \| d_{t}\right)

在这里， :math:`\Phi_{\boldsymbol{z}}` 是目标分布在固定支持 :math:`\boldsymbol{z}` 上的L2投影。

Noisy Net
>>>>>>>>>
Noisy Nets使用一个噪声线性层，它结合了确定性和噪声流：

.. math::

   \boldsymbol{y}=(\boldsymbol{b}+\mathbf{W} \boldsymbol{x})+\left(\boldsymbol{b}_{\text {noisy }} \odot \epsilon^{b}+\left(\mathbf{W}_{\text {noisy }} \odot \epsilon^{w}\right) \boldsymbol{x}\right)

随着时间的推移，网络可以学习忽略噪声流，但在状态空间的不同部分以不同的速率进行学习，从而实现一种自适应的状态条件探索，即一种自退火机制。当动作空间很大时，例如在 Montezuma's Revenge 等游戏中噪声网络通常比 :math:`\epsilon`-greedy 方法取得更好的改进效果，这是由于 :math:`\epsilon`-greedy 往往会在收集足够数量的动作奖励之前迅速收敛到一个 one hot 分布。
在我们的实现中，噪声在每次前向传播时都会重新采样，无论是在数据收集还是训练过程中。当使用双重Q学习时，目标网络也会在每次前向传播之前重新采样噪声。噪声采样过程中，噪声首先从 :math:`N(0,1)` 中进行采样，然后通过一个保持符号的平方根函数进行调节，即 :math:`x \rightarrow x.sign() * x.sqrt()`.

Intergrated Method
>>>>>>>>>>>>>>>>>>

首先，我们将一步的 distributional loss 替换为多步损失：

.. math::

   \begin{split}
   D_{\mathrm{KL}}\left(\Phi_{\boldsymbol{z}} d_{t}^{(n)} \| d_{t}\right) \\
   d_{t}^{(n)}=\left(R_{t}^{(n)}+\gamma_{t}^{(n)} z,\quad p_{\bar{\theta}}\left(S_{t+n}, a_{t+n}^{*}\right)\right)
   \end{split}

然后，我们将多步 distributional loss 与 Double DQN相结合，通过使用在线网络选择贪婪动作，并使用目标网络评估该动作。KL损失也被用来优先选择转换：

.. math::

   p_{t} \propto\left(D_{\mathrm{KL}}\left(\Phi_{z} d_{t}^{(n)} \| d_{t}\right)\right)^{\omega}

网络有共享的表征层, 之后将其输入到 :math:`N_{atoms}` 输出的值流 :math:`v_\eta` 中,以及　 :math:`N_{atoms} \times N_{actions}` 输出的优势函数流 :math:`a_{\psi}`， 在这里 :math:`a_{\psi}^i(a)` 表示与原子i和动作a对应的输出。对于每个原子 :math:`z_i` ， 值流和优势流被聚合，类似于 Dueling DQN，然后通过softmax层进行处理，以获得用于估计回报分布的归一化参数化分布：

.. math::

  \begin{split}
  p_{\theta}^{i}(s, a)=\frac{\exp \left(v_{\eta}^{i}(\phi)+a_{\psi}^{i}(\phi, a)-\bar{a}_{\psi}^{i}(s)\right)}{\sum_{j} \exp \left(v_{\eta}^{j}(\phi)+a_{\psi}^{j}(\phi, a)-\bar{a}_{\psi}^{j}(s)\right)} \\
  \text { where } \phi=f_{\xi}(s) \text { and } \bar{a}_{\psi}^{i}(s)=\frac{1}{N_{\text {actions }}} \sum_{a^{\prime}} a_{\psi}^{i}\left(\phi, a^{\prime}\right)
  \end{split}

扩展
-----------
- Rainbow 可以与以下技术相结合使用:
  
  - 循环神经网络 (RNN)


实现
----------------
Rainbow 默认参数如下:

.. autoclass:: ding.policy.rainbow.RainbowDQNPolicy
   :noindex:

Rainbow使用的网络接口被定义如下：

.. autoclass:: ding.model.template.q_learning.RainbowDQN
   :members: forward
   :noindex:

基准
------------

+---------------------+-----------------+-----------------------------------------------------+--------------------------+----------------------+
| environment         |best mean reward | evaluation results                                  | config link              | comparison           |
+=====================+=================+=====================================================+==========================+======================+
|                     |                 |                                                     |`config_link_p <https://  |                      |
|                     |                 |                                                     |github.com/opendilab/     |  Tianshou(21)        |
|                     |     21          |                                                     |DI-engine/tree/main/dizoo/|                      |
|Pong                 |                 |.. image:: images/benchmark/pong_rainbow.png         |atari/config/serial/      |                      |
|                     |                 |                                                     |pong/pong_rainbow_config  |                      |
|(PongNoFrameskip-v4) |                 |                                                     |.py>`_                    |                      |
+---------------------+-----------------+-----------------------------------------------------+--------------------------+----------------------+
|                     |                 |                                                     |`config_link_q <https://  |                      |
|                     |                 |                                                     |github.com/opendilab/     |  Tianshou(16192.5)   |
|Qbert                |      20600      |                                                     |DI-engine/tree/main/dizoo/|                      |
|                     |                 |.. image:: images/benchmark/qbert_rainbow.png        |atari/config/serial/      |                      |
|(QbertNoFrameskip-v4)|                 |                                                     |qbert/qbert_rainbow_config|                      |
|                     |                 |                                                     |.py>`_                    |                      |
+---------------------+-----------------+-----------------------------------------------------+--------------------------+----------------------+
|                     |                 |                                                     |`config_link_s <https://  |                      |
|                     |                 |                                                     |github.com/opendilab/     |  Tianshou(1794.5)    |
|SpaceInvaders        |     2168        |                                                     |DI-engine/tree/main/dizoo/|                      |
|                     |                 |.. image:: images/benchmark/spaceinvaders_rainbow.png|atari/config/serial/      |                      |
|(SpaceInvadersNoFrame|                 |                                                     |spaceinvaders/spaceinvad  |                      |
|skip-v4)             |                 |                                                     |ers_rainbow_config.py>`_  |                      |
+---------------------+-----------------+-----------------------------------------------------+--------------------------+----------------------+


P.S.:

1. 上述结果是通过在五个不同的随机种子 (0, 1, 2, 3, 4)上运行相同的配置获得的。
2. 对于离散动作空间算法，通常使用 Atari 环境集进行测试(包括子环境 Pong ) ，而 Atari 环境通常通过训练10M个环境步骤的最高平均奖励来评估。有关 Atari 的更多详细信息, 请参阅 `Atari Env Tutorial <../env_tutorial/atari.html>`_ .

关于Rainbow算法的实验技巧
-----------------------------
我们在LunarLander环境上进行了实验， 把 rainbow (dqn) 策略与 n-step, dueling, priority, and priority_IS 等基准比较. 实验的代码链接在这里 `here <https://github.com/opendilab/DI-engine/blob/main/dizoo/box2d/lunarlander/config/lunarlander_dqn_config.py>`_.
请注意，配置文件默认设置为 ``dqn`` ，如果我们想采用 ``rainbow`` 需要将策略类型更改如下：



.. code-block:: python

   lunarlander_dqn_create_config = dict(
    env=dict(
        type='lunarlander',
        import_names=['dizoo.box2d.lunarlander.envs.lunarlander_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='rainbow'),
   )


以下是关于实验设置的详细说明：

+---------------------+---------------------------------------------------------------------------------------------------+
| Experiments setting | Remark                                                                                            |
+=====================+===================================================================================================+
| base                | one step DQN (n-step=1, dueling=False, priority=False, priority_IS=False)                         |
+---------------------+---------------------------------------------------------------------------------------------------+
| n-step              | n step DQN (n-step=3, dueling=False, priority=False, priority_IS=False)                           |
+---------------------+---------------------------------------------------------------------------------------------------+
| dueling             | use dueling head trick (n-step=3, dueling=True, priority=False, priority_IS=False)                |
+---------------------+---------------------------------------------------------------------------------------------------+
| priority            | use priority experience replay buffer (n-step=3, dueling=False, priority=True, priority_IS=False) |
+---------------------+---------------------------------------------------------------------------------------------------+
| priority_IS         | use importance sampling tricks (n-step=3, dueling=False, priority=True, priority_IS=True)         |
+---------------------+---------------------------------------------------------------------------------------------------+




1. ``reward_mean`` 相对于 ``training iteration`` 被作为评估指标。

2. 每个实验设置将使用随机种子0、1和2进行三次运行，并对结果进行平均，以确保结果的随机性。

.. code-block:: python

   if __name__ == "__main__":
      serial_pipeline([main_config, create_config], seed=0)

3. By setting the ``exp_name`` in config file, the experiment results can be saved in specified path. Otherwise, it will be saved in ``‘./default_experiment’`` directory.

.. code-block:: python


   from easydict import EasyDict
   from ding.entry import serial_pipeline

   nstep = 1
   lunarlander_dqn_default_config = dict(
    exp_name='lunarlander_exp/base-one-step2',
    env=dict(
       ......



结果如下图所示。可以看到，通过使用技巧，收敛速度大大加快。在这个实验设置中， Dueling trick 对性能的贡献最大。

.. image::
   images/rainbow_exp.png
   :align: center



参考文献
-----------
**(DQN)** Mnih, Volodymyr, et al. "Human-level control through deep reinforcement learning." 2015; [https://deepmind-data.storage.googleapis.com/assets/papers/DeepMindNature14236Paper.pdf]

**(Rainbow)** Matteo Hessel, Joseph Modayil, Hado van Hasselt, Tom Schaul, Georg Ostrovski, Will Dabney, Dan Horgan, Bilal Piot, Mohammad Azar, David Silver: “Rainbow: Combining Improvements in Deep Reinforcement Learning”, 2017; [http://arxiv.org/abs/1710.02298 arXiv:1710.02298].

**(Double DQN)** Van Hasselt, Hado, Arthur Guez, and David Silver: "Deep reinforcement learning with double q-learning.", 2016; [https://arxiv.org/abs/1509.06461 arXiv:1509.06461]

**(PER)** Schaul, Tom, et al.: "Prioritized Experience Replay.", 2016; [https://arxiv.org/abs/1511.05952 arXiv:1511.05952]

William Fedus, Prajit Ramachandran, Rishabh Agarwal, Yoshua Bengio, Hugo Larochelle, Mark Rowland, Will Dabney: “Revisiting Fundamentals of Experience Replay”, 2020; [http://arxiv.org/abs/2007.06700 arXiv:2007.06700].

**(Dueling network)** Wang, Z., Schaul, T., Hessel, M., Hasselt, H., Lanctot, M., & Freitas: "Dueling network architectures for deep reinforcement learning", 2016; [https://arxiv.org/abs/1511.06581 arXiv:1511.06581]

**(Multi-step)** Sutton, R. S., and Barto, A. G.: "Reinforcement Learning: An Introduction". The MIT press, Cambridge MA. 1998; 

**(Distibutional RL)** Bellemare, Marc G., Will Dabney, and Rémi Munos.: "A distributional perspective on reinforcement learning.", 2017; [https://arxiv.org/abs/1707.06887 arXiv:1707.06887]

**(Noisy net)** Fortunato, Meire, et al.: "Noisy networks for exploration.", 2017; [https://arxiv.org/abs/1706.10295 arXiv:1706.10295]

其他开源实现
>>>>>>>>>>>>>>>>>>>>>>

- `Tianshou <https://github.com/thu-ml/tianshou/blob/master/tianshou/policy/modelfree/rainbow.py>`_

- `RLlib <https://github.com/ray-project/ray/blob/master/rllib/agents/dqn/dqn.py>`_









