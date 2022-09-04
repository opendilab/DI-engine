PLR
^^^^^^^

Overview
---------
PLR was proposed in `Prioritized Level Replay <https://arxiv.org/abs/2010.03934>`_.  PLR is a method for sampling training levels that exploits the differences in learning potential among levels to improve both sample efficiency and generalization.

Quick Facts
-----------
1. PLR supports the **multi-level environments**.

2. PLR updates **policy** and **level score** at the same time.

3. In the implementation of DI-engine, PLR is combined with **PPG** algorithm.

4. PLR supports **Policy entropy**, **Policy min-margin**, **Policy least-confidence**, **1-step TD error** and **GAE** score function. 

Key Graphs
----------
Game levels are determined by a random seed and can vary in navigational layout, visual appearance, and starting positions of entities. 
PLR selectively samples the next training level based on an estimated learning potential of replaying each level anew. The next level is either sampled from a distribution with support over unseen levels (top), which could be the environment’s (perhaps implicit) full training-level distribution, or alternatively, sampled from the replay distribution, which prioritizes levels based on future learning potential (bottom).

.. image:: images/PLR_pic.png
   :align: center
   :height: 250

Key Equations
-------------
The Scoring Levels for Learning Potential is:

.. image:: images/PLR_Score.png
   :align: center
   :height: 250

Given level scores, we use normalized outputs of a prioritization function :math:`h` evaluated over these scores and tuned using a temperature parameter :math:`\beta` to define the score-prioritized distribution :math:`P_{S}\left(\Lambda_{\text {train }}\right)` over the training levels, under which

.. math::

    P_{S}\left(l_{i} \mid \Lambda_{\text {seen }}, S\right)=\frac{h\left(S_{i}\right)^{1 / \beta}}{\sum_{j} h\left(S_{j}\right)^{1 / \beta}}

.. math::
    
    h\left(S_{i}\right)=1 / \operatorname{rank}\left(S_{i}\right)

where :math:`\operatorname{rank}\left(S_{i}\right)` is the rank of level score :math:`S_{i}` among all scores sorted in descending order.

As the scores used to parameterize :math:`P_{S}` are a function of the state of the policy at the time the associated level was last played, they come to reflect a gradually more off-policy measure the longer they remain without an update through replay. We mitigate this drift towards “off-policy-ness” by explicitly mixing the sampling distribution with a staleness prioritized distribution :math:`P_{C}` :

.. math::

    P_{C}\left(l_{i} \mid \Lambda_{\text {seen }}, C, c\right)=\frac{c-C_{i}}{\sum_{C_{j} \in C} c-C_{j}}

.. math::

    P_{\text {replay }}\left(l_{i}\right)=(1-\rho) \cdot P_{S}\left(l_{i} \mid \Lambda_{\text {seen }}, S\right)+\rho \cdot P_{C}\left(l_{i} \mid \Lambda_{\text {seen }}, C, c\right)


Pseudo-code
-----------

Policy-gradient training loop with PLR

.. image:: images/PLR_1.png


Experience collection with PLR

.. image:: images/PLR_2.png


Benchmark
--------------

.. list-table:: Benchmark of PLR algorithm
   :widths: 25 30 15
   :header-rows: 1

   * - environment
     - evaluation results
     - config link
   * - | BigFish
     - .. image:: images/PLR_result.png
     - `config_link_p <https://github.com/opendilab/DI-engine/blob/main/dizoo/procgen/entry/bigfish_plr_config.py>`_

References
-----------

Minqi Jiang, Edward Grefenstette, Tim Rocktaschel: “Prioritized Level Replay”, 2021; arXiv:2010.03934.


Other Public Implementations
------------------------------

- [facebookresearch](https://github.com/facebookresearch/level-replay)
