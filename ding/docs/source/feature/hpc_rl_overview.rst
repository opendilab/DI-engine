HPC_RL Overview
===================



概述
*****
HPC_RL 组件是一个加速算子组件，全称是 High Performance Computation。针对强化学习算法中较通用的算法模块，例如 ``GAE`` ，``n-step TD`` 以及 ``LSTM`` 等，主要是针对DI-engine rl_utils，torch_utils/network, torch_utils/loss下的算子进行优化，算子支持前向+反向传播，训练，数据收集，测试模块中均可使用，对于各类算子，在不同输入尺寸下均有10-100倍的速度提升。

如何使用
*********
    1. 安装

        HPC_RL 目前支持的环境版本是：
          
          - 系统：linux
          - CUDA：CUDA9.2
          - Python：py3.6

        由于 HPC_RL 目前依赖于特定的环境版本，所以我们现在会单独提供对应版本下 HPC_RL 组件打包好的 whl 文件，可通过 ``pip install <whl_name>`` 安装。在 PROJECT_PATH/ding/hpc_rl 目录下有我们已经提供好的 whl 文件可供安装。

        安装成功后，在python终端中如果可以成功 ``import hpc_rl`` ，则说明安装成功

        .. tip::

            使用新版本时，可能会出现某些不匹配问题，建议删除老版本后重新安装新版本，如安装目录在 ``~/.local/lib/python3.6/site-packages`` 下，则执行如下的命令即可删除：

            .. code:: bash

                rm ~/.local/lib/python3.6/site-packages/hpc_*.so
                rm ~/.local/lib/python3.6/site-packages/hpc_rl* -rf
    2. 验证

       当安装成功后，使用者可以通过DI-engine/hpc_rl/tests下的单元测试来验证精度和效率，这些测试会运行原始版本基于pytorch api的实现+ HPC_RL优化后的实现，例如运行该目录下的test_gae.py，在 ``Tesla V100 32G`` 上的运行结果如下：

       .. code:: bash

            target problem: T = 1024, B = 64
            gae mean_relative_error: -1.0645836e-07
            ================run gae performance test================
            epoch: 0, original gae cost time: 0.06686019897460938
            epoch: 1, original gae cost time: 0.06580924987792969
            epoch: 2, original gae cost time: 0.0658106803894043
            epoch: 3, original gae cost time: 0.06581401824951172
            epoch: 4, original gae cost time: 0.06805300712585449
            epoch: 5, original gae cost time: 0.06583261489868164
            epoch: 0, hpc gae cost time: 0.0024187564849853516
            epoch: 1, hpc gae cost time: 0.0024328231811523438
            epoch: 2, hpc gae cost time: 0.0034339427947998047
            epoch: 3, hpc gae cost time: 0.0014312267303466797
            epoch: 4, hpc gae cost time: 0.0024368762969970703
            epoch: 5, hpc gae cost time: 0.002432107925415039

    3. 使用

        DI-engine中默认关闭HPC_RL的使用（因为目前仅支持部分运行环境），若成功安装后，可在入口程序最开始处加上一行代码 ``ding.enable_hpc_rl = True`` ，即会自动启用HPC_RL相关算子，demo如下：

        .. code:: python

            import ding
            from ding.entry import serial_pipeline
            from dizoo.classic_control.cartpole.config.cartpole_a2c_config import cartpole_a2c_config, cartpole_a2c_create_config


            if __name__ == "__main__":
                ding.enable_hpc_rl = True
                cartpole_a2c_config.policy.cuda = True
                serial_pipeline([cartpole_a2c_config, cartpole_a2c_create_config], seed=0)

    4. demo

        在qbert上使用dqn算法时，在 ``main.py`` 中设置 ``ding.enable_hpc_rl = True``，可以看到训练时间从9.7ms降低到8.8ms。运行平台是16GV100，CUDA9.2。

        Pytorch:

        +-------+----------------+------------+----------------+
        | Name  | train_time_val | cur_lr_val | total_loss_val |
        +-------+----------------+------------+----------------+
        | Value | 0.008813       | 0.000100   | 0.008758       |
        +-------+----------------+------------+----------------+

        HPC_RL:
        
        +-------+----------------+------------+----------------+
        | Name  | train_time_val | cur_lr_val | total_loss_val |
        +-------+----------------+------------+----------------+
        | Value | 0.009722       | 0.000100   | 0.426298       |
        +-------+----------------+------------+----------------+


目前支持的算子
****************
   ``rl_utils`` : GAE, PPO, q_value n-step TD, dist n_step TD(C51), q_value n-step TD with rescale(R2D2)，TD-lambda, vtrace, UPGO

   ``torch_utils/network`` : LSTM，scatter_connection

性能对比
********

    .. csv-table:: Performance on Pytorch and HPC_RL
        :header: "operator", "shape", "test environment", "pytorch", "HPC_RL"
        :widths: 30, 80, 60, 40, 40

        "TD-lambda", "T=16, B=16", "32GV100, CUDA9.2", "900us", "95us"
        "TD-lambda", "T=256, B=64", "32GV100, CUDA9.2", "13.1ms", "105us"
        "TD-lambda", "T=256, B=512", "32GV100, CUDA9.2", "18.8ms", "130us"
        "dntd", "T=16, B=128, N=128", "32GV100, CUDA10.1", "2000us", "424us"
        "dntd", "T=128, B=16, N=128", "32GV100, CUDA10.1", "5860us", "420us"
        "dntd", "T=128, B=128, N=16", "32GV100, CUDA10.1", "5930us", "422us"
        "dntd", "T=128, B=128, N=128", "32GV100, CUDA10.1", "5890us", "420us"
        "dntd", "T=512, B=128, N=128", "32GV100, CUDA10.1", "19120us", "423us"
        "dntd", "T=128, B=128, N=512", "32GV100, CUDA10.1", "5940us", "463us"
        "gae", "T=16, B=16", "32GV100, CUDA10.1", "1110us", "36us"
        "gae", "T=16, B=64", "32GV100, CUDA10.1", "1150us", "36us"
        "gae", "T=256, B=64", "32GV100, CUDA10.1", "15510us", "82us"
        "gae", "T=256, B=256", "32GV100, CUDA10.1", "15730us", "83us"
        "gae", "T=1024, B=16", "32GV100, CUDA10.1", "62810us", "235us"
        "gae", "T=1024, B=64", "32GV100, CUDA10.1", "65850us", "240us"
        "lstm", "seq_len=16, B=4", "32GV100, CUDA10.1", "50969us", "8311us"
        "lstm", "seq_len=64, B=4", "32GV100, CUDA10.1", "204976us", "29383us"
        "lstm", "seq_len=64, B=16", "32GV100, CUDA10.1", "204073us", "25769 us"
        "lstm", "seq_len=256, B=4", "32GV100, CUDA10.1", "845367us", "113733us"
        "lstm", "seq_len=256, B=16", "32GV100, CUDA10.1", "861429us", "98873us"
        "ppo", "B=16, N=16", "32GV100, CUDA10.1", "2037us", "388us"
        "ppo", "B=16, N=128", "32GV100, CUDA10.1", "2047us", "389us"
        "ppo", "B=128, N=16", "32GV100, CUDA10.1", "2032us", "389us"
        "ppo", "B=128, N=128", "32GV100, CUDA10.1", "2153us", "394us"
        "ppo", "B=512, N=128", "32GV100, CUDA10.1", "2143us", "393us"
        "ppo", "B=512, N=512", "32GV100, CUDA10.1", "2047us", "3898us"
        "qntd", "T=16, B=128, N=128", "32GV100, CUDA10.1", "1248us", "254us"
        "qntd", "T=128, B=16, N=128", "32GV100, CUDA10.1", "5429us", "261us"
        "qntd", "T=128, B=128, N=16", "32GV100, CUDA10.1", "5214us", "253us"
        "qntd", "T=128, B=128, N=128", "32GV100, CUDA10.1", "5179us", "257us"
        "qntd", "T=512, B=128, N=128", "32GV100, CUDA10.1", "18355us", "254us"
        "qntd", "T=128, B=128, N=512", "32GV100, CUDA10.1", "5198us", "254us"
        "qntd_rescale", "T=16, B=128, N=128", "32GV100, CUDA10.1", "1655us", "266us"
        "qntd_rescale", "T=128, B=16, N=128", "32GV100, CUDA10.1", "5652us", "264us"
        "qntd_rescale", "T=128, B=128, N=16", "32GV100, CUDA10.1", "5653us", "265us"
        "qntd_rescale", "T=128, B=128, N=128", "32GV100, CUDA10.1", "5653us", "265us"
        "qntd_rescale", "T=512, B=128, N=128", "32GV100, CUDA10.1", "19286us", "264us"
        "qntd_rescale", "T=128, B=128, N=512", "32GV100, CUDA10.1", "5677us", "265us"
        "scatter", "B=16, M=64, N=64", "32GV100, CUDA10.1", "559us", "311us"
        "scatter", "B=64, M=16, N=64", "32GV100, CUDA10.1", "561us", "309us"
        "scatter", "B=64, M=64, N=16", "32GV100, CUDA10.1", "567us", "310us"
        "scatter", "B=64, M=64, N=64", "32GV100, CUDA10.1", "571us", "309us"
        "scatter", "B=256, M=64, N=64", "32GV100, CUDA10.1", "852us", "480us"
        "scatter", "B=256, M=64, N=256", "32GV100, CUDA10.1", "2399us", "1620us"
        "upgo", "T=16, B=128, N=128", "32GV100, CUDA10.1", "2274us", "247us"
        "upgo", "T=128, B=16, N=128", "32GV100, CUDA10.1", "13350us", "246us"
        "upgo", "T=128, B=128, N=16", "32GV100, CUDA10.1", "13367us", "246us"
        "upgo", "T=128, B=128, N=128", "32GV100, CUDA10.1", "13421us", "269us"
        "upgo", "T=512, B=128, N=128", "32GV100, CUDA10.1", "51923us", "749us"
        "upgo", "T=128, B=128, N=512", "32GV100, CUDA10.1", "13705us", "474us"
        "vtrace", "T=16, B=128, N=128", "32GV100, CUDA10.1", "2906us", "325us"
        "vtrace", "T=128, B=16, N=128", "32GV100, CUDA10.1", "10979us", "328us"
        "vtrace", "T=128, B=128, N=16", "32GV100, CUDA10.1", "10906us", "368us"
        "vtrace", "T=128, B=128, N=128", "32GV100, CUDA10.1", "11095us", "459us"
        "vtrace", "T=512, B=128, N=128", "32GV100, CUDA10.1", "39693us", "1364us"
        "vtrace", "T=128, B=128, N=512", "32GV100, CUDA10.1", "12230us", "776us"

其他
*********

1. 为了提升性能，HPC_RL在内部默认会预先分配算子所需要的内存，因此需要知道数据的具体尺寸，DI-engine的相关wrapper会自动根据数据尺寸进行调整，但要注意，如果是可变输入尺寸，反复重新分配内存会造成一定的时间损耗，从而降低加速比。
2. 对于部分算子，例如当映射关系有重叠时，GPU上并行执行，映射结果是不确定的，会存在一定的数值精度波动，但基本不影响常规训练。
3. 对于部分算子，HPC_RL只支持其中某些常见的参数组合，具体如下：

  - q_value n-step TD 的 criterion 仅支持MSE
  - q_value n-step TD with rescale 的 criterion 仅支持MSE，trans_fn, inv_trans_fn仅支持R2D2中的相关变换形式
  - LSTM中的normalization仅支持LN
