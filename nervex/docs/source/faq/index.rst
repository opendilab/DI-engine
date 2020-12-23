FAQ
=====================

.. toctree::
   :maxdepth: 2

Q1: 如何使用串行版本入口
************************

:A1:
  如下面所示的代码，串行训练入口一共有五个参数

.. code:: python 

    def serial_pipeline(
            cfg: Union[str, dict],
            seed: int,
            env_setting: Optional[Any] = None,  # subclass of BaseEnv, and config dict
            policy_type: Optional[type] = None,  # subclass of Policy
            model_type: Optional[type] = None,  # subclass of torch.nn.Module
    ) -> None:
        pass


- cfg: 该参数可以是一个字符串，即配置文件的路径（绝对路径相对路径均可），或是一个dict，即已经加载处理完成的配置dict
- seed: 该参数是随机种子，为一个int值，会设置各类外部库的随机种子，以及环境的随机种子，注意多个环境跟根据环境序号再加上相应数字作为种子，以保证不同环境种子不同
- env_setting(optional): 该参数用来设置环境，一般为None，即从全局配置文件中创建环境，否则是一个list，其中有三个元素，第一个元素是环境类，第二第三个参数分别是环境的配置dict的list，各自的长度等于需要创建的环境个数。
- policy_type(optional): 该参数用来设置Policy，一般为None，即从全局配置文件中创建策略，当用户实现了自己的policy时，可以通过相应的注册机制注册进入nervex，从而可以通过配置文件方式调用，也可以通过该参数直接将新定义的策略类传进来。
- model_type(optional): 该参数用来设置神经网络模型，一般为None，nervex已实现的策略使用的默认的神经网络，用户可以通过该参数传入自己定义的神经网络。

Q2: 如何自定义环境
********************

:A2:
   - 需要继承 ``nervex/envs/env/base_env.py`` 中的BaseEnv类，按照基类的相关说明和示例实现相应的方法。
   - 注意环境中只是推荐使用tensor作为输入输出的数据类型，用户也可使用numpy数组或是python内置类型，在接入各类训练pipeline时，可以通过指定EnvManager实例创建时的tensor_transformer参数为真，让其自动完成环境数据到tensor的转换。nervex系统中除环境之外的部分，一律使用tensor作为基本数据类型，其他类型会导致运行错误。
   - 环境step方法返回的info **必须** 为dict，且当一个episode结束时，info中 **必须** 包括 ``final_eval_reward`` 这个键，其将作为评价整个episode性能的指标，要求 ``final_eval_reward`` 的取值为python内置数据类型(int, float)
   - (optional)如果想要在nervex中通过配置文件使用自定义的环境，需要在自定义环境文件中全局范围调用 ``register_env`` 方法进行注册，并在配置文件中指定环境名(env_type)，以及加载的模块名(import_names)
   - (optional)BaseEnv的info方法返回环境相关的参数信息，但其并不会和系统其他模块强耦合，只是作为 **可选** 的一个方法，使用者也可自定义相关格式，以及在系统中最终的用法
   - (optional) create_actor_env_cfg和create_evaluator_env_cfg方法会解析传入的配置文件，生成相应的环境配置list，该方法的目的是处理不同环境使用不同配置文件的情况，如果用户不想使用该方法，可以使用串行入口的env_setting参数进行信息传递。


Q3: 如何自定义神经网络
************************

:A3:
  - 如果不使用nervex中Agent, Adder等抽象模块，完全自定义实现策略，则神经网络只需要继承torch.nn.Module实现即可，没有任何其他要求。
  - 如果使用上述模块，则神经网络的输入输出一般情况下 **必须** 为dict，且dict中相关键值对的键必须满足一些限制，具体可以参考agent部分的相关文档。


Q4: 如何自定义Policy(策略)
*****************************

:A4:
  - 需要继承 ``nervex/policy/base_policy.py`` 中的Policy类，按照基类的相关说明和示例实现相应的方法。
  - (optional)如果想要在nervex中通过配置文件使用自定义的策略，需要在自定义环境文件中全局范围调用 ``register_policy`` 方法进行注册，并在配置文件中指定策略名(policy_type)，以及加载的模块名(import_names)


Q5: 配置文件中的traj_len, unroll_len概念
*****************************************

:A5:
  - unroll_len指的是训练端，每次迭代展开的步数长度，一般情况下为1，在有RNN（处理时间序列的神经网络）的情况下才会大于1，该数值如果不为1的话，应该尽可能设置的大一点，使得用来学习的序列尽量长，一般最大值是根据显存和RNN的训练收益进行调整。
  - traj_len是actor部分一个trajectory的长度，一个trajectory是一次数据发送的基本单元，一般大于等于unroll_len。在不考虑数据吞吐效率的情况下，尽可能设置大一点，如果设置为字符串 ``inf`` ，那么就是以一个完整的episode来发送数据，trajectory越长，某些操作（比如GAE）可以传递的reward值就越远，有利于训练收敛。为了平衡效率，一般设置为unroll_len的整数倍。
  - 如果使用nstep return，traj_len = n * unroll_len + nstep，其中n = 1, 2, 3, ... , 从而保证traj最后几帧数据也能有效利用。
  - 如果使用GAE，traj_len > 1
