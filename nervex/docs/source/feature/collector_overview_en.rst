Collector Overview
====================

Profile of Speed
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We conduct speed tests of our collector in comparison with Tianshou, under environments with different scales and environment managers with different types.

**As the results show, our collector generally outperforms Tianshou's, especially under environments with large scales or long reset times.**

- Settings

    We design three environments with different scales, whose parameters are as follows:

        +------------------------+---------------+--------------+--------------+-------------+
        |                        |Observation Dim| Reset Time   |  Step Time   |  Infer Time |
        +========================+===============+==============+==============+=============+
        |       Small Env        |      64       |     0.1      |     0.005    |     0.004   |
        +------------------------+---------------+--------------+--------------+-------------+
        |      Middle Env        |      300      |     0.5      |     0.01     |     0.008   |
        +------------------------+---------------+--------------+--------------+-------------+
        |         Big Env        |      3000     |       2      |      0.1     |     0.02    |
        +------------------------+---------------+--------------+--------------+-------------+

    where Observation Dim means the dimension of the observation, Reset Time, Step Time and Infer Time are the times the collector needs to reset the environment, execute a step of the environment and use the policy to infer a step. In our tests, the three types of Time uniformly fluctuate in ``[0.4, 1.6]`` times of their original values in the above table.


    We conducts tests under sync and async modes, where the collector runs **8** environments in the environment manager. Under sync mode, we set ``wait_num`` as 8 and the collector waits for all **8** environments, while under async mode, it is **7**. We retain the default values of ``timeout`` for both our and Tianshou's  env managers due to the different implementations of the async mode.

        +------------------------+---------------+--------------------+
        |                        |    Wait Num   |    Timeout         |
        +========================+===============+====================+
        |         Sync           |      8        |     None           |
        +------------------------+---------------+--------------------+
        |         Async          |      7        | 0.01(nx) / None(ts)|
        +------------------------+---------------+--------------------+

    Basically, for each environment, we conduct 4 tests with the env managers under sync and async modes in ours and Tianshou, namely **ours-sync, ours-async, ts-sync, ts-async**. Additionally, we include our base env manager as baseline, namely **ours-base**.

    We simulate **300** collections during one test with each collection producing at least **80** samples, and repeat each test 3 times to get the averages and standard variances. To approximate the real-world scenes, we run a pure cpu task to raise the utilization of cpu to ~60% before each test.

- Results

    The results of speed tests are as follows(unit: second):

        +------------------------+---------------+--------------+--------------+-------------+-------------+
        |                        |    ts-async   |   ts-sync    |   ours-base  |  ours-async |  ours-sync  |
        +========================+===============+==============+==============+=============+=============+
        |       Small Env        |  49.54+0.35   |  44.63+0.09  | 157.70+0.30  | 47.60+0.62  | 47.19+1.13  |
        +------------------------+---------------+--------------+--------------+-------------+-------------+
        |      Middle Env        |  93.02+0.09   |  88.70+0.14  | 311.88+0.22  | 90.84+0.67  | 82.73+1.51  |
        +------------------------+---------------+--------------+--------------+-------------+-------------+
        |         Big Env        | 545.07+1.55   | 520.52+0.30  | 2592.77+0.25 | 519.05+1.05 | 467.50+2.18 |
        +------------------------+---------------+--------------+--------------+-------------+-------------+

        We conclude the findings as:

            1. The speeds of our subprocess collector is 3~5 times of the base collector's.
            2. Our collector is faster than Tianshou's in all of the six settings except the Small Env under sync mode.

    To approximate some environments with large reset times, e.g. Carla and StarCraft2, we further conduct a group of tests with all reset times multiplied by **5** (unit: second):

        +------------------------+---------------+--------------+--------------+-------------+-------------+
        |                        |    ts-async   |   ts-sync    |    nx-base   |   nx-async  |   nx-sync   |
        +========================+===============+==============+==============+=============+=============+
        |       Small Env        |  55.11+0.10   |  45.10+0.08  | 176.55+0.05  | 50.71+0.80  | 50.55+1.39  |
        +------------------------+---------------+--------------+--------------+-------------+-------------+
        |      Middle Env        | 130.49+0.09   | 112.27+0.03  | 407.65+0.14  | 98.49+1.29  | 94.18+1.74  |
        +------------------------+---------------+--------------+--------------+-------------+-------------+
        |         Big Env        | 703.49+0.61   | 577.92+0.30  | 2976.80+0.39 | 555.15+1.90 | 520.31+1.05 |
        +------------------------+---------------+--------------+--------------+-------------+-------------+

        As shown above, Tianshou's collector bears drastic speed decreases in such environments compared with our collector, which uses reset threads to avoid busy waiting and remain high performance under large reset times.
