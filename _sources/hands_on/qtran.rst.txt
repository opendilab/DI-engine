QTRAN
^^^^^^^

Overview
---------
QTRAN is proposed by Kyunghwan et al.(2019). QTRAN is a factorization method for MARL, which is free from such structural constraints and takes on a new approach to transform the original joint action-value function into an easily factorizable one, with the same optimal actions.

Compared to VDN(Sunehag et al. 2017), QMIX(Rashid et al. 2018), QTRAN guarantees more general factorization than VDN or QMIX, thus covering a much wider class of MARL tasks than does previous methods, and it performs better than QMIX in 5m_vs_6m and MMM2 maps.

Quick Facts
-------------
1. QTRAN uses the paradigm of **centralized training with decentralized execution**.

2. QTRAN is a **model-free** and **value-based** method.

3. QTRAN only support **discrete** action spaces.

4. QTRAN is an **off-policy multi-agent** RL algorithm.

5. QTRAN considers a **partially observable** scenario in which each agent only obtains individual observations.

6. QTRAN accepts **DRQN** as individual value network.

7. QTRAN learns the **joint value function** through **Individual action-value network**, **Joint action-value network** and **State-value network**.

Key Equations or Key Graphs
---------------------------
The overall QTRAN architecture including individual agent networks and the mixing network structure:

.. image:: images/marl/Qtran_net.png
   :align: center
   :width: 600

QTRAN trains the mixing network via minimizing the following loss:

.. math::
   L_{\mathrm{td}}(; \boldsymbol{\theta}) =\left(Q_{\mathrm{jt}}(\boldsymbol{\tau}, \boldsymbol{u})-y^{\mathrm{dqn}}\left(r, \boldsymbol{\tau}^{\prime} ; \boldsymbol{\theta}^{-}\right)\right)^{2}

.. math::
   L_{\mathrm{opt}}(; \boldsymbol{\theta}) =\left(Q_{\mathrm{jt}}^{\prime}(\boldsymbol{\tau}, \overline{\boldsymbol{u}})-\hat{Q}_{\mathrm{jt}}(\boldsymbol{\tau}, \overline{\boldsymbol{u}})+V_{\mathrm{jt}}(\boldsymbol{\tau})\right)^{2}

.. math::
   L_{\mathrm{nopt}}(; \boldsymbol{\theta}) =\left(\min \left[Q_{\mathrm{jt}}^{\prime}(\boldsymbol{\tau}, \boldsymbol{u})-\hat{Q}_{\mathrm{jt}}(\boldsymbol{\tau}, \boldsymbol{u})+V_{\mathrm{jt}}(\boldsymbol{\tau}), 0\right]\right)^{2}

.. math::
   L\left(\boldsymbol{\tau}, \boldsymbol{u}, r, \boldsymbol{\tau}^{\prime} ; \boldsymbol{\theta}\right)=L_{\mathrm{td}}+\lambda_{\mathrm{opt}} L_{\mathrm{opt}}+\lambda_{\mathrm{nopt}} L_{\mathrm{nopt}},

Pseudo-code
-----------
The following flow charts show how QTRAN trains.

.. image:: images/marl/QTRAN.png
   :align: center
   :width: 600

Extensions
-----------
- QTRAN++ (Son et al. 2019), as an extension of QTRAN, successfully bridges the gap between empirical performance and theoretical guarantee, and newly achieves state-of-the-art performance in the SMAC environment.

Implementations
----------------
The default config is defined as follows:

    .. autoclass:: ding.policy.qmix.QTRANPolicy
        :noindex:

The network interface QTRAN used is defined as follows:
    .. autoclass:: ding.model.template.qtran
        :members: forward
        :noindex:

The Benchmark result of QTRAN in SMAC (Samvelyan et al. 2019), for StarCraft micromanagement problems, implemented in DI-engine is shown.


+---------------------+-----------------+-----------------------------------------------------+--------------------------+----------------------+
| smac map            |best mean reward | evaluation results                                  | config link              | comparison           |
+=====================+=================+=====================================================+==========================+======================+
|                     |                 |                                                     |`config_link <https://    |                      |
|                     |                 |                                                     |github.com/opendilab/     |  Pymarl(1.0)         |
|                     |                 |                                                     |DI-engine/tree/main/dizoo/|                      |
|MMM                  |  1.00           |.. image:: images/benchmark/QTran_MMM.png            |smac/config/              |                      |
|                     |                 |                                                     |smac_MMM_qtran_config     |                      |
|                     |                 |                                                     |.py>`_                    |                      |
+---------------------+-----------------+-----------------------------------------------------+--------------------------+----------------------+
|                     |                 |                                                     |`config_link <https://    |                      |
|                     |                 |                                                     |github.com/opendilab/     |  Pymarl(0.1)         |
|3s5z                 |                 |                                                     |DI-engine/tree/main/dizoo/|                      |
|                     |  0.95           |.. image:: images/benchmark/QTran_3s5z.png           |smac/config/              |                      |
|                     |                 |                                                     |smac_3s5z_qtran_config    |                      |
|                     |                 |                                                     |.py>`_                    |                      |
+---------------------+-----------------+-----------------------------------------------------+--------------------------+----------------------+
|                     |                 |                                                     |`config_link <https://    |                      |
|                     |                 |                                                     |github.com/opendilab/     |  Pymarl(0.7)         |
|5m6m                 |                 |                                                     |DI-engine/tree/main/dizoo/|                      |
|                     |  0.55           |.. image:: images/benchmark/QTran_5m6m.png           |smac/config/              |                      |
|                     |                 |                                                     |smac_3s5z_qtran_config    |                      |
|                     |                 |                                                     |.py>`_                    |                      |
+---------------------+-----------------+-----------------------------------------------------+--------------------------+----------------------+



References
----------------
QTRAN: Learning to Factorize with Transformation for Cooperative Multi-Agent Reinforcement Learning. ICML, 2019.


Other Public Implementations
-----------------------------
- `Pymarl <https://github.com/oxwhirl/pymarl>`_.
