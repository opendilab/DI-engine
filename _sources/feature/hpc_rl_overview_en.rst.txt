DI-hpc Overview
===================



Overview
************
The HPC_RL component (High Performance Computation) is an acceleration operator component for general algorithm modules in reinforcement learning algorithms, such as ``GAE``, ``n-step TD`` and ``LSTM``, etc., mainly for the operators in ding ``rl_utils``, ``torch_utils/network`` and ``torch_utils/loss``. The operators support forward and backward propagation, and can be used in training, data collection, and test modules. For all types of operators, there are 10- A speed increase of 100 times.

How to use
************
    1. Installation

        The environment version currently supported by HPC_RL is:
          
            - System: linux
            - CUDA: CUDA9.2
            - Python: py3.6

        Since HPC_RL currently depends on a specific environment version, we will now provide the .whl file of the HPC_RL component under the corresponding version separately, which can be installed through ``pip install <whl_name>``. In the PROJECT_PATH/ding/hpc_rl directory, there is a whl file that we have provided for installation.

        If you can successfully ``import hpc_rl`` in the python terminal, the installation is successful

        .. tip::

            When using the new version, some mismatch problems may occur. It is recommended to delete the old version and reinstall the new version. If the installation directory is under ``~/.local/lib/python3.6/site-packages``, execute The following command can be deleted:

            .. code:: bash

                rm ~/.local/lib/python3.6/site-packages/hpc_*.so
                rm ~/.local/lib/python3.6/site-packages/hpc_rl* -rf

    2. Verification

        After the installation, users can verify the accuracy and efficiency through the unit tests under ding/hpc_rl/tests. These tests will run the original version based on the pytorch api implementation and the HPC_RL optimized implementation. For example, run the ``test_gae.py`` in this directory, the results of running on ``Tesla V100 32G`` are as follows:

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

    3. Usage

        The usage of HPC_RL is disabled by default in ding (because only part of the operating environment is currently supported). After installation, you can add a line of code ``ding.enable_hpc_rl = True`` at the beginning of the entry program, and HPC_RL related calculations will be automatically enabled. The demo is as follows:

        .. code:: python

            import ding
            from ding.entry import serial_pipeline
            from dizoo.classic_control.cartpole.config.cartpole_a2c_config import cartpole_a2c_config, cartpole_a2c_create_config


            if __name__ == "__main__":
                ding.enable_hpc_rl = True
                cartpole_a2c_config.policy.cuda = True
                serial_pipeline([cartpole_a2c_config, cartpole_a2c_create_config], seed=0)



    4. Demo

        We provide a demo on qbert using dqn algorithm. With setting ``ding.enable_hpc_rl = True`` in ``main.py``, the training time will drop from 9.7ms to 8.8ms on 16GV100 with CUDA9.2.

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


Currently supported operators
******************************
    ``rl_utils`` : GAE, PPO, q_value n-step TD, dist n_step TD(C51), q_value n-step TD with rescale(R2D2)，TD-lambda, vtrace, UPGO

    ``torch_utils/network`` : LSTM，scatter_connection


Performance comparison
***********************

    .. csv-table:: Performance between Pytorch and HPC_RL
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


Others
*********

1. In order to improve performance, HPC_RL will pre-allocate the memory required by the operator internally by default, so you need to know the specific size of the data. The relevant wrapper of ding will automatically adjust according to the data size, but note that if it is a variable input size , Repeated reallocation of memory will cause a certain amount of time loss, thereby reducing the speedup.

2. For some operators, for example, when the mapping relationship overlaps, they are executed in parallel on the GPU, and the mapping result is uncertain, and there will be certain numerical accuracy fluctuations, but it basically does not affect conventional training.

3. For some operators, HPC_RL only supports some common parameter combinations, as follows:

   - q_value n-step TD criterion only supports MSE
   - The criterion of q_value n-step TD with rescale only supports MSE, trans_fn, inv_trans_fn only support the relevant transformation form in R2D2
   - Normalization in LSTM only supports LN
