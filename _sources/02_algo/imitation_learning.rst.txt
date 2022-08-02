Imitation Learning
====================

Problem Definition and Research Motivation
--------------------------------------------

Imitation Learning (IL) generally refers to a large class of learning methods in which an agent extracts knowledge from expert data and then imitates the behavior contained in these expert data. Due to the inherent characteristics of IL. It has two main characteristics: it usually requires a large amount of training data, and generally requires that the quality of the training data is good enough. In general, IL can be divided into three directions: IRL (inverse reinforcement learning), BC (behavioral cloning), Adversarial Structured IL, Below we briefly analyze each research direction in this field.



Research Direction
--------------------

Behavioral Cloning (BC)
~~~~~~~~~~~~~~~~~~~~~~~~~

BC was first proposed in [1], which proposes a supervised learning method, which directly establishes the state-action mapping relationship by fitting expert data.

The biggest advantage of BC is that it is simple and efficient, but once the agent encounters some never-before-seen state, it may make fatal mistakes - a problem called "state distribution drift". To solve this problem, DAgger [2] proposed a method to dynamically update the dataset: collect the real state-action pairs encountered with the policy currently being trained, and add these new expert data to the dataset for subsequent policy update. In a recent study, IBC [3] proposed implicit action cloning, the key of which is that the neural network accepts both observations and actions, and outputs a energy value that is low for expert actions and high for non-expert actions, thereby turning behavioral cloning into an energy-based modeling problem.

The current research hotspots of BC algorithms mainly focus on two aspects: meta-learning and behavior cloning using VR devices.


Inverse Reinforcement Learning (IRL)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Inverse reinforcement learning (IRL) is the problem of inferring the reward function of an agent, given its policy or observed behavior. Specifically, IRL first learns a reward function from expert data, and then uses this reward function for subsequent RL training. IRL can theoretically outperform expert data.

From the specific work above, Ziebart et al. [4] first proposed maximum entropy IRL, which utilizes the maximum entropy distribution to better characterize multimodal behavior for more efficient optimization. In 2016, Finn et al. [5] proposed a model-based approach to IRL called guided cost learning, capable of learning arbitrary nonlinear cost functions, such as neural networks, without meticulous feature engineering, and formulate an efficient sample-based approximation for MaxEnt IOC. Subsequently, Hester et al. proposed DQfD [6], which requires only a small amount of expert data, and significantly accelerates the training process through pre-training and a specially designed loss function. T-REX [7] propose a novel reward-learning-from-observation algorithm, that extrapolates beyond a set of (approximately) ranked demonstrations in order to infer high-quality reward functions from a set of potentially poor demonstrations.


Adversarial Structured IL
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The main goal of the Adversarial Structured IL approach is to improve the efficiency of IRL. Even if the IRL algorithm learns a very good reward function, in order to get the final near-optimal policy, it still needs to perform a reinforcement learning step. If the policy can be learned directly from the expert data, the efficiency can be greatly improved. Based on this idea, GAIL [8] combines generative network (GAN) and maximum entropy IRL to learn approximate optimal policies without human annotated expert data.

On this basis, many works have made related improvements to GAIL. For example, InfoGail [9] replaced GAN with WGAN and achieved better performance. There are also some recent works such as GoalGAIL [10], TRGAIL [11] and DGAIL [12] which combine other methods such as post-hoc relabeling and DDPG to achieve faster convergence and better final performance.


Future Study
--------------

There are still many challenges in imitation learning, mainly including the following:

- Generally speaking, it is for a specific task, and there is a lack of imitation learning methods that can be applied to multiple tasks;

- For situations where the data is not optimal, it is difficult to surpass the data to achieve optimal results;

- Mainly focus on the research of observation, without combining multi-modal factors such as speech and natural language;

- The local optimum can be found, but the global optimum can often not be found.

Reference
-----------

.. [1] Michael Bain and Claude Sammut. 1999. A framework for behavioural cloning. In *Machine Intelligence 15*. Oxford University Press, 103-129.

.. [2] Stéphane Ross, Geoffffrey Gordon, and Drew Bagnell. 2011. A reduction of imitation learning and structured prediction to no-regret online learning. In *Proceedings of the fourteenth international conference on artifificial intelligence and* *statistics*. JMLR Workshop and Conference Proceedings, 627-635.

.. [3] Florence, P. , Lynch, C. , Zeng, A. , Ramirez, O. , Wahid, A. , & Downs, L. , et al. (2021). Implicit behavioral cloning.

.. [4] Brian D Ziebart, Andrew L Maas, J Andrew Bagnell, and Anind K Dey. 2008. Maximum entropy inverse reinforcement learning.. In *Aaai*, Vol. 8. Chicago, IL, USA, 1433-1438.

.. [5] Chelsea Finn, Sergey Levine, and Pieter Abbeel. 2016. Guided cost learning: Deep inverse optimal control via policy optimization. In *International conference on machine learning*. PMLR, 49-58.

.. [6] Todd Hester, Matej Vecerik, Olivier Pietquin, Marc Lanctot, Tom Schaul, Bilal Piot, Dan Horgan, John Quan, Andrew Sendonaris, Gabriel Dulac-Arnold, Ian Osband, John Agapiou, Joel Z. Leibo, and Audrunas Gruslys. 2017. Deep Q learning from Demonstrations. *arXiv:1704.03732 [cs]* (Nov. 2017). http://arxiv.org/abs/1704.03732 arXiv: 1704.03732.

.. [7] Daniel Brown, Wonjoon Goo, Prabhat Nagarajan, and Scott Niekum. 2019. Extrapolating beyond suboptimal demonstrations via inverse reinforcement learning from observations. In *International Conference on Machine Learning*. PMLR, 783-792.

.. [8] Jonathan Ho and Stefano Ermon. 2016. Generative Adversarial Imitation Learning. In *Advances in Neural Information* *Processing Systems 29*, D. D. Lee, M. Sugiyama, U. V. Luxburg, I. Guyon, and R. Garnett (Eds.). Curran Associates, Inc., 4565-4573. http://papers.nips.cc/paper/6391-generative-adversarial-imitation-learning.pdf

.. [9] Yunzhu Li, Jiaming Song, and Stefano Ermon. 2017. InfoGAIL: Interpretable Imitation Learning from Visual Demonstrations. In *Advances in Neural Information Processing Systems 30*, I. Guyon, U. V. Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett (Eds.). Curran Associates, Inc., 3812-3822. http://papers.nips.cc/paper/6971-infogail-interpretable-imitation-learning-from-visual-demonstrations.pdf

.. [10] Yiming Ding, Carlos Florensa, Mariano Phielipp, and Pieter Abbeel. 2019. Goal-conditioned imitation learning. *arXiv* *preprint arXiv:1906.05838* (2019).

.. [11] Akira Kinose and Tadahiro Taniguchi. 2020. Integration of imitation learning using GAIL and reinforcement learning using task-achievement rewards via probabilistic graphical model. *Advanced Robotics* (June 2020), 1-13. https://doi.org/10.1080/01691864.2020.1778521

.. [12] Guoyu Zuo, Kexin Chen, Jiahao Lu, and Xiangsheng Huang. 2020. Deterministic generative adversarial imitation learning. *Neurocomputing* 388 (May 2020), 60-69. https://doi.org/10.1016/j.neucom.2020.01.016
