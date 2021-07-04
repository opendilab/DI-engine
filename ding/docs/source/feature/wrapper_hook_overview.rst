Wrapper & Hook Overview
==========================


Wrapper
--------------------
概述：
    Wrapper，即装饰器。一般来说，当我们希望在某个函数执行的时候额外执行一些自定义的操作时，Wrapper 就可以被派上用场。用 Wrapper 对函数进行包装，可以方便地对函数的输入输出进行操作，或者是计算函数相关的一些状态。对于 model 方面的操作，例如 ``.cuda()`` 或者 train/eval 模式切换以及不同 mode 下是否共享模型本身，交给用户在 policy 中直接对 model 进行操作。

用处：
    DI-engine 中用到 wrapper 的地方有三个，分别是 env，model，以及 learner

    - env

        env 里面用到的 wrapper，实际上就是 ``gym.Wrapper`` 的子类。为了方便地对环境类的输入输出做一些操作或者适配，Wrapper 是非常方便且有效的工具。可以简单地理解为，这部分的 Wrapper 是对环境类的一个包装。env_wrapper 中只对常用的 `gym` 库的一些 wrapper 做了封装。

        - 使用：
            .. code:: python

                env = gym.make('PongNoFrameskip-v4')
                env = NoopResetEnv(env)
            
        - 定义自己的 env wrapper, 对于用户自定义的 ``MyWrapper``，需要完成以下几步（与使用 ``gym.Wrapper`` 完全一致）：

            1. ``MyWrapper`` 继承 ``gym.Wrapper``，依据需求实现其中的 ``step``, ``reset`` 等函数
            2. 使用 ``env = MyWrapper(env)`` 来得到新的经过包装的环境

    - model

        对于 policy 中使用的 model，我们对其也实现了和 ``gym.Wrapper`` 相似的封装，以实现对 ``model`` 类更为快速方便的更改。

        - 使用：

            已经定义好的 wrapper 统一放在 ``ding.model.model_wrappers.py`` 下以方便查看。对于使用 wrapper，可以按照如下规则得到新的model：
            
            .. code:: python

                model = model_wrap(model, wrapper_name='your_wrapper_name', **kwargs)

            * wrapper 可以是所需要使用的任何 wrapper
                * 自定义 wrapper 使用参考下一节
            * ``wrapper_name`` 为已经注册的任意 wrapper 的名称。如果是自定义的 wrapper，注册的时候需要提供名称。
            * ``kwargs`` 部分为该 wrapper 所需要的参数
            * 在此情况下，得到的 ``model`` 可以像原本的 model 那样去使用。例如，当调用 ``model.forward`` 的时候，会优先调用 wrapper 中定义的 ``forward`` 函数。如果没有定义的话，会到下一层的 wrapper 中继续寻找。

        - 定义自己的 model wrapper：

            对于用户自定义的 ``MyWrapper``，需要完成以下几步：

            1. 继承 ``ding.model.model_wrappers.IModelWrapper``，该类是 model 所使用的 wrapper 的基类。
            
            2. 在 ``MyWrapper`` 中，依据需求实现所需要的 forward 等函数。
            
            3. 将 ``MyWrapper`` 通过 ``register_wrapper()`` 的方法添加到 ``model_wrappers.wrapper_name_map`` 这个字典中。如此一来，便可以通过 ``add_wrapper`` 方便地对 model 进行添加 wrapper 的操作。
        
        - 调用流程：

            .. image:: images/wrapper_structure.jpg

            .. image:: images/wrapper_call.jpg

        - 目前已经支持的 wrapper：

            .. csv-table:: 
                :header: "Wrapper Name", "Wrapper Class Name", "Wrapper Usage"
                :widths: 50, 50, 60

                "base", "BaseModelWrapper", "最基础的wrapper，提供简单的reset方法"
                "hidden_state", "HiddenStateWrapper", "控制 ``forward`` 时隐状态的行为，在实例内部根据训练batch样本数维护对应的隐状态，每次 ``forward`` 前输入上一次迭代的输出隐状态，而 ``forward`` 后保存该次的输出隐状态为下一次做准备"
                "argmax_sample", "ArgmaxSampleWrapper", "对于 logit 输入，找到最大值所在的的 index，作为动作。用于离散动作"
                "eps_greedy_sample", "EpsGreedySampleWrapper", "对于 q value 输入，利用Epsilon贪婪策略采样动作。用于离散动作"
                "multinomial_sample", "MultinomialSampleWrapper", "对于 logit 输入，根据概率采样动作。用于离散动作"
                "action_noise", "ActionNoiseWrapper", "为动作加上指定种类（如高斯、OU）的噪声。用于连续动作"
                "target", "TargetNetworkWrapper", "用于实现 target network"
                "teacher", "TeacherNetworkWrapper", "用于实现 teacher network"

        - 查看Wrapper嵌套情况

            调用最外层的model.info()方法即可看到所有当前model所添加的wrapper嵌套情况。
            
            .. code:: python


                model = MLP()
                model = model_wrap(model, wrapper_name='multinomial_sample')
                model = model_wrap(model, wrapper_name='argmax_sample')
                print(model.info('forward')) # 查看forward方法在model中的调用情况
                # MultinomialSampleWrapper ArgmaxSampleWrapper MLP 依次打印出forward方法调用情况

    - learner
        model 中用到 wrapper 的地方比较少，主要表现为计时相关的 ``time wrapper``。


Hook
--------------------
概述：
    Hook，钩子，可以通过在钩子内使得外部函数在被调用的时候，自动调用钩子内定义好的函数。在程序中，对于一段封装得较好的代码，如果需要修改的话，也许要花费相当的精力。Hook 函数就是由此被创造出来的。代码作者可以在一段代码中的任意位置暴露出钩子，而用户可以在钩子中实现自己所需要的功能，这样当代码运行到指定位置的时候，钩子会被触发，钩子中定义好的函数会被自动调用，从而实现快速修改代码的功能。
用处：
    DI-engine 中使用 hook 主要是在 learner 中。

    - learner

        在DI-engine中，learner 的训练部分可以简化如下：

        .. code:: python

            # before_run
            for i in range(max_iter):
                # before_iter
                self._policy.forward(data)
                # after_iter
            # after_run

        从代码可以看出，learner 里面用到的 hook 定义了四个位置，分别为

        * before_run：训练任务开始之前
        * after_run：训练任务完成之后
        * before_iter：在训练任务的每个 iter 之前
        * after_iter：在训练任务的每个 iter 之后

        当程序运行到指定位置的时候，在此位置注册的 hook 上的所有函数将会被调用。

        - 使用：

            DI-engine 已经实现了许多常用的 hook，并提供了简单的调用方法。可以通过 cfg 去调用 hook，cfg 配置与使用如下：
            
            .. code:: python

                # hook:
                #     load_ckpt:
                #         name: load_ckpt # hook的名称
                #         position: before_run # hook所处的位置
                #         priority: 20    # hook的优先级，如果同一位置被注册了多个hook，则会根据优先级大小来顺序执行
                #         type: load_ckpt # hook的类型
                #     log_show:
                #         ext_args:
                #             freq: 100   # 提供了参数freq来控制hook被执行的频率
                #         name: log_show
                #         position: after_iter
                #         priority: 20
                #         type:log_show
                #     save_ckpt_after_iter:
                #         ext_args:
                #             freq: 100
                #         name: save_ckpt_after_iter
                #         position: after_iter
                #         priority: 20
                #         type: save_ckpt
                #     save_ckpt_after_run:
                #         name: save_ckpt_after_run
                #         position: after_run
                #         priority: 20
                #         type: save_ckpt
                hooks = build_learner_hook_by_cfg(cfg)

            至此，DI-engine 在初始化 learner 的时候会自动根据 cfg 的内容进行 hook 注册，以保证相关功能能够正常进行。

        - 定义自己的 hook, 对于用户自定义的 ``MyHook``，需要完成以下几步：

            1. 继承 ``ding.worker.learner.learner_hook.LearnerHook``。该类是所有 learner 中使用的 hook 的基类。
            2. 在 ``MyHook`` 中实现 ``__call__`` 方法。``__call__`` 方法的输入是一个 learner 的实例。通过该实例，hook 可以对 learner 中的任意变量进行操作。
            3. 调用 ``register_learner_hook()`` 对自定义的 ``MyHook`` 进行注册，需要提供 hook 名称。
            4. 现在已经可以在 cfg 中使用自定义的 ``MyHook`` 了。

        - 调用流程：

            .. image:: images/hook_call.jpg

        - 目前已经支持的 hook：

            .. csv-table:: 
                :header: "Hook Name", "Hook Class Name", "Hook Usage"
                :widths: 50, 50, 60

                "load_ckpt", "LoadCkptHook", "读取模型的检查点"
                "save_ckpt", "SaveCkptHook", "保存模型到检查点"
                "log_show", "LogShowHook", "打印日志"
                "log_reduce", "LogReduceHook", "集合处理日志"

        - 简化调用的 hook：

            由于前面提到的 hook 存在参数复杂，不利于初学者上手等原因，DI-engine 提供了更为简单的调用方法：

            .. csv-table:: Simplified Hook in DI-engine
                :header: "Hook Name", "Params", "Hook Usage"
                :widths: 50, 50, 60

                "log_show_after_iter", "freq", "根据参数给定的freq每隔一定数量个iter之后打印日志"
                "load_ckpt_before_run", "None", "在训练程序运行之前读取检查点"
                "save_ckpt_after_iter", "freq", "根据参数给定的freq每隔一定数量个iter之后保存模型"
                "save_ckpt_after_run", "None", "在训练程序运行完全之后保存模型"

            调用方法也更为简单，通过下面的代码即可得到所需 hooks:
            
            .. code:: python

                hook_cfg = dict(
                    save_ckpt_after_iter=20, # 在 after_iter 位置添加了名称为 save_ckpt 的 hook，每隔20个iter会存一次ckpt
                    save_ckpt_after_run=True, # 在 after_run 位置添加了名称为 save_ckpt 的 hook，训练完毕的时候会存一次ckpt
                ) 
                hooks = build_learner_hook_by_cfg(hook_cfg)

        - 查看 hook 调用情况：

            DI-engine 提供了 ``show_hooks()`` 方法以便查看各个位置的 hook 调用情况，具体如下：
            
            .. code:: python  

                from ding.worker.learner.learner_hook import show_hooks
                from ding.worker.learner import build_learner_hook_by_cfg
                cfg = dict(save_ckpt_after_iter=20, save_ckpt_after_run=True)
                hooks = build_learner_hook_by_cfg(cfg)
                show_hooks(hooks)
                # before_run: []
                # after_run: ['SaveCkptHook']
                # before_iter: []
                # after_iter: ['SaveCkptHook']

.. note::
    Wrapper 和 Hook 的区别？

    * Wrapper 是对原始函数的封装，支持一层一层的复用，如果在当前层没有找到对应的函数方法，会在更上一层去寻找。
    * Hook 是在原始方法的基础上，在某个位置插入一个新的方法。
    
    .. image:: images/wrapper_hook_call.jpg


