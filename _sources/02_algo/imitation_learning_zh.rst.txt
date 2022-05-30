模仿学习
==================

问题与意义
----------

模仿学习 (Imitation Learning, IL) 指的是，智能体通过学习一些专家数据来提取知识，进而复制下这些专家数据的行为这样一种学习方法。由于IL的本身特性，
它面临两大难题：需要大量的训练数据、训练数据的质量一定要好。为了解决上述问题，从大体来说，IL可以分成三个方向：IRL（逆强化学习），BC（行为克隆），
Adversarial Structured IL（对抗结构）。下面对各个方向做简要分析：

研究方向
--------

BC
~~~~~~~~

BC 最早提出于[1]，它提出了一种监督学习的方法，通过拟合专家数据，直接建立状态-动作的映射关系。

BC 的最大好处是效率很高，算法简单，但是一旦智能体遇到了从未见过的状态，就可能做出错误的行为——这一问题被称作“状态漂移”。为了解决这个问题，DAgger[2]方法采用了一种动态更新数据集的方法，根据训练出 policy 遇到的真实状态，不断添加新的专家数据至数据集中。而在后续的研究中，IBC[3] 采用了隐式行为克隆的方法，它的关键是训练一个神经网络来接受观察和动作，并输出一个数字，该数字对专家动作来说很低，对非专家动作来说很高，从而将行为克隆变成一个基于能量的建模问题。

目前的 BC 算法研究热点主要聚焦于两个方面：meta-learning 和利用 VR 设备进行行为克隆。

IRL
~~~~~~~~

IRL 的主要目标是为了解决数据收集时，难以找到足够高质量数据的问题。具体来说，IRL 首先从专家数据中学习一个奖励函数，进而使用这个奖励函数进行后续的RL训练。通过这样的方法，IRL 从理论上来说，可以表现出超越专家数据的性能。

从具体的工作上面，Ziebart等人[4] 首先提出了最大熵 IRL，它利用最大熵分布来获得良好的前景和有效的优化。后来在2016年，Finn等人[5]提出了一种基于模型的 IRL 方法，称为引导成本学习（
guided cost
learning），这种方法使用神经网络表示 cost 进而提高表达能力。后续，Hester等人又提出了DQfD[6]，该方法仅需少量的专家数据，通过预训练启动过程和后续学习过程，显著加速了训练。后来的方法如 T-REX[7] 提出了一种基于为专家数据排序的结构，通过对比什么专家数据效果更好，间接地学习奖励函数。

Adversarial Structured IL
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Adversarial Structured IL 方法的主要目标是为了解决 IRL 的效率问题。通过 IRL 的算法可以看出，即便它学到了非常好的奖励函数，由于得到最终的策略仍然需要执行强化学习步骤，因此如果可以直接从专家数据中学习到策略，就可以大大提高效率。基于这个想法 GAIL
[8] 结合了生成式网络 GAN 和最大熵 IRL，无需人工不断标注专家数据，就可以不断地高效训练。

在此基础上，许多工作都对 GAIL 做了改进。如 InfoGail
[9]用 WGAN 替换了 GAN，取得了较好的效果。还有一些近期的工作，如 GoalGAIL[10]，TRGAIL[11] 和 DGAIL[12] 都结合了其他方法，如事后重标记和 DDPG，以实现更快的收敛速度和更好的最终性能。

未来展望
--------

当前模仿学习还存在许多挑战，主要包括以下几点：

- 当前的模仿学习都是针对某个特定任务而言的，缺乏能适用于多任务的模仿学习方法；

- 当前模仿学习算法对于专家数据并非最优的情形，难以超越专家数据达到最优结果；

- 当前的模仿学习算法主要针对 observation 的，没有能结合语音、自然语言等多模态因素；

- 当前模仿学习能够找到局部的最优点，但往往不能找到全局的最优点。

参考文献
--------

[1] Michael Bain and Claude Sammut. 1999. A framework for behavioural
cloning. In *Machine Intelligence 15*. Oxford

University Press, 103–129.

[2] Stéphane Ross, Geoffffrey Gordon, and Drew Bagnell. 2011. A
reduction of imitation learning and structured prediction to no-regret
online learning. In *Proceedings of the fourteenth international
conference on artifificial intelligence and*

*statistics*. JMLR Workshop and Conference Proceedings, 627–635.

[3] Florence, P. , Lynch, C. , Zeng, A. , Ramirez, O. , Wahid, A. , &
Downs, L. , et al. (2021). Implicit behavioral cloning.

[4] Brian D Ziebart, Andrew L Maas, J Andrew Bagnell, and Anind K Dey.
2008. Maximum entropy inverse reinforcement

learning.. In *Aaai*, Vol. 8. Chicago, IL, USA, 1433–1438.

[5] Chelsea Finn, Sergey Levine, and Pieter Abbeel. 2016. Guided cost
learning: Deep inverse optimal control via policy

optimization. In *International conference on machine learning*. PMLR,
49–58.

[6] Todd Hester, Matej Vecerik, Olivier Pietquin, Marc Lanctot, Tom
Schaul, Bilal Piot, Dan Horgan, John Quan, Andrew

Sendonaris, Gabriel Dulac-Arnold, Ian Osband, John Agapiou, Joel Z.
Leibo, and Audrunas Gruslys. 2017. Deep Q learning from Demonstrations.
*arXiv:1704.03732 [cs]* (Nov. 2017). http://arxiv.org/abs/1704.03732
arXiv: 1704.03732.

[7] Daniel Brown, Wonjoon Goo, Prabhat Nagarajan, and Scott Niekum.
2019. Extrapolating beyond suboptimal demon

strations via inverse reinforcement learning from observations. In
*International Conference on Machine Learning*.

PMLR, 783–792.

[8] Jonathan Ho and Stefano Ermon. 2016. Generative Adversarial
Imitation Learning. In *Advances in Neural Information*

*Processing Systems 29*, D. D. Lee, M. Sugiyama, U. V. Luxburg, I.
Guyon, and R. Garnett (Eds.). Curran Associates, Inc.,

4565–4573.
http://papers.nips.cc/paper/6391-generative-adversarial-imitation-learning.pdf

[9] Yunzhu Li, Jiaming Song, and Stefano Ermon. 2017. InfoGAIL:
Interpretable Imitation Learning from Vi

sual Demonstrations. In *Advances in Neural Information Processing
Systems 30*, I. Guyon, U. V. Luxburg,

S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett
(Eds.). Curran Associates, Inc., 3812–3822.

http://papers.nips.cc/paper/6971-infogail-interpretable-imitation-learning-from-visual-demonstrations.pdf

[10] Yiming Ding, Carlos Florensa, Mariano Phielipp, and Pieter Abbeel.
2019. Goal-conditioned imitation learning. *arXiv*

*preprint arXiv:1906.05838* (2019).

[11] Akira Kinose and Tadahiro Taniguchi. 2020. Integration of imitation
learning using GAIL and reinforcement

learning using task-achievement rewards via probabilistic graphical
model. *Advanced Robotics* (June 2020), 1–13.

https://doi.org/10.1080/01691864.2020.1778521

[12] Guoyu Zuo, Kexin Chen, Jiahao Lu, and Xiangsheng Huang. 2020.
Deterministic generative adversarial imitation

learning. *Neurocomputing* 388 (May 2020), 60–69.
https://doi.org/10.1016/j.neucom.2020.01.016
