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
            model: Optional[Union[type, torch.nn.Module]] = None,  # instance or subclass of torch.nn.Module
            enable_total_log: Optional[bool] = False,
    ) -> None:
        pass


- cfg: 该参数可以是一个字符串，即配置文件的路径（绝对路径相对路径均可），或是一个dict，即已经加载处理完成的配置dict
- seed: 该参数是随机种子，为一个int值，会设置各类外部库的随机种子，以及环境的随机种子，注意多个环境跟根据环境序号再加上相应数字作为种子，以保证不同环境种子不同
- env_setting(optional): 该参数用来设置环境，一般为None，即从全局配置文件中创建环境，否则是一个list，其中有三个元素，第一个元素是环境类型，第二个参数是一个list，其中元素的值是某个进行数据收集的环境的配置dict，list的总长度是进行数据收集的环境综述，第三个参数类似第二个参数，只是环境的功能变为评测性能。
- policy_type(optional): 该参数用来设置Policy，一般为None，即从全局配置文件中创建策略，当用户实现了自己的policy时，可以通过相应的注册机制注册进入nervex，从而可以通过配置文件方式调用，也可以通过该参数直接将新定义的策略类传进来。
- model(optional): 该参数用来设置神经网络模型，一般为None，nervex已实现的策略使用的默认的神经网络，用户可以通过该参数传入自己定义的神经网络(支持直接传入模型实例或是传入模型类型，再通过配置文件中的model字段完成创建)。
- enable_total_log(optional): 如果该参数为真，会开启nervex所有模块所有级别的log，默认为假，即默认关闭部分log

Q2: 如何自定义环境
********************

:A2:
   - 需要继承 ``nervex/envs/env/base_env.py`` 中的BaseEnv类，按照基类的相关说明和示例实现相应的方法。
   - 环境推荐使用numpy.ndarray作为array的数据类型，即环境反馈给外界，外界输入环境的数据都以np数组为基本类型。在环境管理器中，这些np数据通过BaseEnvManager实例中的变换函数self._transform和self._inv_transform变换为nervex内部使用的数据，默认为PyTorch Tensor。
     self._transform从外部数据类型往环境数据转换类型，self_inv_transform从环境数据类型往外部数据类型转换。nervex系统中除环境之外的部分，一律使用tensor作为基本数据类型，其他类型会导致运行错误。
   - 环境step方法返回的info **必须** 为dict，且当一个episode结束时，info中 **必须** 包括 ``final_eval_reward`` 这个键，其将作为评价整个episode性能的指标，要求 ``final_eval_reward`` 的取值为python内置数据类型(int, float)
   - (optional)如果想要在nervex中通过配置文件使用自定义的环境，需要在自定义环境文件中全局范围调用 ``nervex/utils/registry_factory/ENV_REGISTRY`` 这一 ``Registry`` 模块的实例进行注册，可使用装饰器或调用函数两类方法进行注册。并在配置文件中指定环境名(``env_type``)，以及加载的模块名(``import_names``)
   - (optional)BaseEnv的info方法返回环境相关的参数信息，但其并不会和系统其他模块强耦合，只是作为 **可选** 的一个方法，使用者也可自定义相关格式，以及在系统中最终的用法
   - (optional) create_collector_env_cfg和create_evaluator_env_cfg方法会解析传入的配置文件，生成相应的环境配置list，该方法的目的是处理不同环境使用不同配置文件的情况，如果用户不想使用该方法，可以使用串行入口的env_setting参数进行信息传递。


Q3: 如何自定义神经网络
************************

:A3:
  - 如果不使用nervex中Wrapper, Adder等抽象模块，完全自定义实现策略，则神经网络只需要继承torch.nn.Module实现即可，没有任何其他要求。
  - 如果使用上述模块，则神经网络的输入输出一般情况下 **必须** 为dict，且dict中相关键值对的键必须满足一些限制，具体可以参考wrapper部分的相关文档。


Q4: 如何自定义Policy(策略)
*****************************

:A4:
  - 需要继承 ``nervex/policy/base_policy.py`` 中的Policy类，按照基类的相关说明和示例实现相应的方法。
  - (optional)如果想要在nervex中通过配置文件使用自定义的策略，需要在自定义环境文件中全局范围调用 ``register_policy`` 方法进行注册，并在配置文件中指定策略名(policy_type)，以及加载的模块名(import_names)


Q5: 配置文件中的traj_len, unroll_len概念
*****************************************

:A5:
  - unroll_len指的是训练端，每次迭代展开的步数长度，一般情况下为1，在有RNN（处理时间序列的神经网络）的情况下才会大于1，该数值如果不为1的话，应该尽可能设置的大一点，使得用来学习的序列尽量长，一般最大值是根据显存和RNN的训练收益进行调整。
  - traj_len是collector部分一个trajectory的长度，一个trajectory是一次数据发送的基本单元，一般大于等于unroll_len。在不考虑数据吞吐效率的情况下，尽可能设置大一点，如果设置为字符串 ``inf`` ，那么就是以一个完整的episode来发送数据，trajectory越长，某些操作（比如GAE）可以传递的reward值就越远，有利于训练收敛。为了平衡效率，一般设置为unroll_len的整数倍。
  - 如果使用nstep return，traj_len = n * unroll_len + nstep，其中n = 1, 2, 3, ... , 从而保证traj最后几帧数据也能有效利用。
  - 如果使用GAE，traj_len > 1

Q6: 如何加载/保存模型
**********************

:A6:
 - 加载模型：只需指定配置文件中的 ``load_path`` 字段即可，该字段默认为 ``''`` ，即为不加载模型，如需要加载指定具体的绝对路径即可。
 - 保存模型：对于串行版本，系统默认有两种保存模型的情形，一是当前 ``eval_reward`` 大于等于训练目标 ``stop_value`` ，保存最终的模型并关闭整个训练模块，二是当前 ``eval_reward`` 大于之前最高的reward，则会保存当前的模型。使用者也可以在配置文件中添加相应的 ``save_ckpt`` hook，即每隔一定迭代数保存模型。对于并行版本，默认保存最新的模型用于通信，使用者也可类似添加hook。
 - 具体添加 ``load_path`` 和 ``save_ckpt`` hook可以参见 ``app_zoo/classic_control/cartpole/entry/cartpole_dqn_default_config.yaml``

Q7: 关于使用时出现的warning
****************************

:A7:

对于运行nervex时命令行中显示的import linlink, ceph, memcache, redis的相关warning，一般使用者忽略即可，nervex会在import时自动进行寻找相应的替代库。


Q8: 训练完成后如何评测模型的性能
*********************************

:A8:

在训练完成之后，可在 ``log/evaluator/evalautor_logger.txt`` 中看到训练过程中的评测结果及对应保存的checkpoint名称(``ckpt_name``)，nervex也提供了简单的接口进行评测，流程如下：
 - 准备一个待评测的checkpoint，一般为 ``torch.save`` 保存的文件，内部结构为一个dict，其中 ``model`` 键所指的为神经网络模型权重
 - 准备一个评测用的配置文件，大部分内容和训练配置文件相同，只需添加 ``learner.load_path`` 字段为checkpoint的绝对路径
 - 在shell脚本中运行 ``nervex -m eval -c <config_path> -s <seed>`` 即可，如需指定其他参数，可以调用 ``nervex.entry.application_entry`` 中的eval函数。
 - 如果需要把环境评测的过程保存成视频文件，需要环境实现 ``enable_save_replay`` 接口，并指定配置文件中 ``env.replay_path`` 字段，会将视频文件存储在 ``replay_path`` 目录下


Q9: 如何设置subprocess_env_manager的相关运行参数
**************************************************

:A9:

在配置文件的env字段添加manager字段，可以指定是否使用shared_memory，多进程multiprocessing启动的上下文，具体示例可参考cartpole相关配置文件


Q10: 安装之后无法使用nervex命令行工具(CLI)
********************************************

:A10:

- 部分环境使用pip安装时指定 ``-e`` 选项会导致无法使用CLI，一般非开发者无需指定该选项，去掉该选项重新安装即可
- 部分环境会将CLI安装在用户目录下，需要验证CLI的安装目录是否在使用者的环境变量中


Q11: 安装时出现"没有权限"相关错误
***********************************

:A11:

由于某些运行环境中缺少相应权限，pip安装时可能出现"没有权限"(Permission denied)，具体原因及解决方法如下：
 - pip添加 ``--user`` 选项，安装在用户目录下
 - 将仓库根目录下的 ``.git`` 文件夹移动出去，执行pip安装命令，再将其移动回来，具体原因可参见 https://github.com/pypa/pip/issues/4525
