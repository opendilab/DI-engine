GAIL 
====

Overview
--------

GAIL (Generative Adversarial Imitation Learning) was first proposed in
`Generative Adversarial Imitation
Learning <https://arxiv.org/abs/1606.03476>`__, which deduced the
optimization objective of GAIL from the perspective of occupancy
measure. Compared to other learning methods, GAIL neither suffers from
the compounding error problem in imitation learning, nor needs to
expensively learn the inter-mediate reward function as in inverse
reinforcement learning. But similar to other methods, GAIL is also
exposed to "the curse of dimensionality", which makes the scalability
much valuable in high-dimension-space problems.

Quick Facts
-----------

1. GAIL consists of a generator and a discriminator, trained in an
   adversarial manner.

2. The generator is optimized for a surrogate reward, usually by
   policy-gradient reinforcement learning methods, like TRPO, for its
   sampling nature.

3. The discriminator can be simply optimized by typical gradient descent
   methods, like Adam, to distinguish expert and generated data.

Key Equations or Key Graphs
---------------------------

The objective function in GAIL's adversarial training is as below:

.. figure:: images/gail_loss.png
   :align: center


where \pi is the generator policy, D is the discriminator policy,
while :math:`H(\pi)` is the causal entropy of policy \pi. This is a
min-max optimization process, and the objective is optimized in an
iterative adversarial manner.

Pseudo-Code
-----------

.. figure:: images/GAIL.png
   :alt: 

Extensions
----------

-  MAGAIL (Multi-Agent Generative Adversarial Imitation Learning)

   Multi-agent systems tend to be much more complicated, due to the
   heterogeneity, stochasticity, and interaction among multi-agents.

   `MAGAIL：Multi-Agent Generative Adversarial Imitation
   Learning <https://arxiv.org/abs/1807.09936>`_ extended GAIL to
   multi-agent scenarios. The generator is redefined as a policy
   controlling all agents in a distributed manner, while the
   discriminator is distinguishing expert and generated behavior for
   each agent.

   The Pseudo-Code is as following:

   .. figure:: images/MAGAIL.png
      :scale: 85%
      :alt: 

-  Other perspective to understand GAIL

   GAIL is closely related to other learning methods, and thus can be
   understood in different views.

   `A Connection Between Generative Adversarial Networks, Inverse
   Reinforcement Learning, and Energy-Based
   Models <https://arxiv.org/abs/1611.03852>`__ indicated GAIL's
   implicit connection to GAN, IRL, and energy-based probability
   estimation.

Reference
---------

1. Ho, Jonathan, and Stefano Ermon. "Generative adversarial imitation
   learning." Advances in neural information processing systems 29
   (2016): 4565-4573.

2. Song, Jiaming, et al. "Multi-agent generative adversarial imitation
   learning." arXiv preprint arXiv:1807.09936 (2018).

3. Finn, Chelsea, et al. "A connection between generative adversarial
   networks, inverse reinforcement learning, and energy-based models."
   arXiv preprint arXiv:1611.03852 (2016).
