GTrXL
^^^^^^^

Overview
---------
Gated Transformer-XL, or GTrXL, first proposed in `Stabilizing Transformers for Reinforcement Learning <https://arxiv.org/pdf/1910.06764.pdf>`_,
is a novel framework for reinforcement learning adapted from the Transformer-XL architecture.
It mainly introduces two architectural modifications that improve the stability and learning speed of Transformer including:
placing the layer normalization on only the input stream of the submodules, and replacing residual connections with gating layers.
The proposed architecture, surpasses LSTMs on challenging memory environments and achieves state-of-the-art
results on the several memory benchmarks, exceeding the performance of an external memory architecture.

Quick Facts
-------------
1. GTrXL can serve as a **backbone** for many RL algorithms.

2. GTrXL only supports **sequential** observations.

3. GTrXL is based on **Transformer-XL** with **Gating connections**.

4. The DI-engine implementation of GTrXL is based on the R2D2 algorithm. In the original paper, it is based on the algorithm `V-MPO <https://arxiv.org/abs/1909.12238>`_.

Key Equations or Key Graphs
---------------------------
**Transformer-XL**: to address the context fragmentation problem, Transformer-XL introduces the notion of recurrence to the deep self-attention network.
Instead of computing the hidden states from scratch for each new segment, Transformer-XL reuses the hidden states obtained in previous segments.
The reused hidden states serve as memory for the current segment, which builds up a recurrent connection between the segments.
As a result, modeling very long-term dependency becomes possible because information can be propagated through the recurrent connections.
In order to enable state reuse without causing temporal confusion, Transformer-XL proposes a new relative positional encoding formulation that generalizes to attention lengths longer than the one observed during training.

.. image:: images/transformerXL_train_eval.png
   :align: center

**Identity Map Reordering**: move the layer normalization to the input stream of the submodules.
A key benefit to this reordering is that it now enables an identity map from the input of the transformer at the first layer to the output of the transformer after the last layer.
This is in contrast to the canonical transformer, where there are a series of layer normalization operations that non-linearly transform the state encoding.
One hypothesis as to why the Identity Map Reordering improves results is as follows: assuming that the submodules at initialization produce values that are in expectation near
zero, the state encoding is passed un-transformed to the policy and value heads, enabling the agent to learn a Markovian policy at the start of training
(i.e., the network is initialized such that :math:`\pi(·|st,...,s1) ≈ \pi(·|st)` and :math:`V^\pi(s_t|s_{t-1},...,s_1) ≈ V^\pi(s_t|s_{t-1})`),
thus ignoring the contribution of past observations coming from the memory of the attention-XL.
In many environments, reactive behaviours need to be learned before memory-based ones can be effectively utilized.
For example, an agent needs to learn how to walk before it can learn how to remember where it has walked.
With identity map reordering the forward pass of the model can be computed as:

.. image:: images/identity_map_reordering.png
   :align: center

.. image:: images/gtrxl.png
   :align: center
   :height: 300

**Gating layers**: replace the residual connections with gating layers. Among several studied gating functions, Gated Recurrent Unit (GRU) is the one that performs the best.
Its adapted powerful gating mechanism can be expressed as:

.. math::
   \begin{aligned}
   r &= \sigma(W_r^{(l)} y + U_r^{(l)} x) \\
   z &= \sigma(W_z^{(l)} y + U_z^{(l)} x - b_g^{(l)}) \\
   \hat{h} &= \tanh(W_g^{(l)} y + U_g^{(l)} (r \odot x)) \\
   g^{(l)}(x, y) &= (1-z)\odot x + z\odot \hat{h}
   \end{aligned}

**Gated Identity Initialization**: the authors claimed that the Identity Map Reordering aids policy optimization because it initializes the agent close to a Markovian policy or value function.
If this is indeed the cause of improved stability, we can explicitly initialize the various gating mechanisms to be close to the identity map.
This is the purpose of the bias :math:`b_g^{(l)}` in the applicable gating layers. The authors demonstrate in an ablation that initially setting :math:`b_g^{(l)}>0` produces the best results.

Extensions
-----------
GTrXL can be combined with:

   - CoBERL (`CoBERL: Contrastive BERT for Reinforcement Learning <https://arxiv.org/pdf/2107.05431.pdf>`_):

      Contrastive BERT (CoBERL) is a reinforcement learning agent that combines a new contrastive loss and a hybrid LSTM-transformer
      architecture to tackle the challenge of improving data efficiency for RL. It uses bidirectional masked prediction in combination
      with a generalization of recent contrastive methods to learn better representations for transformers in RL, without the need of hand engineered data augmentations.

   - R2D2 (`Recurrent Experience Replay in Distributed Reinforcement Learning <https://openreview.net/pdf?id=r1lyTjAqYX>`_):

     Recurrent Replay Distributed DQN (R2D2) demonstrates how replay and the RL learning objective can be adapted to work well for agents with recurrent
     architectures. The LSTM can be replaced or combined with gated transformer so that we can leverage the benefits of distributed experience collection, storing
     the recurrent agent state in the replay buffer, and "burning in" a portion of the unrolled network with replayed sequences during training.

Implementations
----------------
The network interface GTrXL used is defined as follows:

.. autoclass:: ding.torch_utils.network.gtrxl.GTrXL
   :members: reset_memory, get_memory, forward
   :noindex:

The default implementation of our R2D2-based GTrXL is defined as follows:

.. autoclass:: ding.policy.r2d2_gtrxl.R2D2GTrXLPolicy
   :members: _forward_learn, _data_preprocess_learn, _init_learn
   :noindex:


Benchmark
-----------


+---------------------+-----------------+-----------------------------------------------------+--------------------------+----------------------+
| environment         |best mean reward | evaluation results                                  | config link              | comparison           |
+=====================+=================+=====================================================+==========================+======================+
|                     |                 |                                                     |`config_link_p <https://  |                      |
|                     |                 |                                                     |github.com/opendilab/     |                      |
|                     |                 |                                                     |DI-engine/tree/main/dizoo/|                      |
|Pong                 |  20             |.. image:: images/benchmark/pong_gtrxl_r2d2.png      |atari/config/serial/      |                      |
|                     |                 |                                                     |pong/pong_dqn_config      |                      |
|(PongNoFrameskip-v4) |                 |                                                     |.py>`_                    |                      |
+---------------------+-----------------+-----------------------------------------------------+--------------------------+----------------------+


P.S.：

1. The above results are obtained by running the same configuration on five different random seeds (0, 1, 2, 3, 4)
2. For the discrete action space algorithm like DQN, the Atari environment set is generally used for testing (including sub-environments Pong), and Atari environment is generally evaluated by the highest mean reward training 10M ``env_step``. For more details about Atari, please refer to `Atari Env Tutorial <../env_tutorial/atari.html>`_ .


Reference
----------

- Parisotto, Emilio, et al. "Stabilizing Transformers for Reinforcement Learning.", 2019; [http://arxiv.org/abs/1910.06764 arXiv:1910.06764]

- Dai, Zihang , et al. "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context.", 2019; [http://arxiv.org/abs/1901.02860 arXiv:1901.02860]



