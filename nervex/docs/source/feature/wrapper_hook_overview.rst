Wrapper & Hook Overview
===================


Wrapper
--------------------
概述：
    Wrapper，即装饰器。一般来说，当我们希望在某个函数执行的时候额外执行一些自定义的操作时，Wrapper就可以被派上用场。用Wrapper对函数进行包装，可以方便地对函数的输入输出进行操作，或者是计算函数相关的一些状态。

用处：
    nervex中用到wrapper的地方有三个，分别是 env，model，以及learner
    - env
        env里面用到的wrapper，实际上就是 `gym.Wrapper` 的子类。为了方便地对环境类的输入输出做一些操作或者适配，Wrapper 是非常方便且有效的工具。可以简单地理解为，这部分的Wrapper是对环境类的一个包装。env_wrapper中只对常用的gym的一些wrapper做了封装。
        - 定义自己的 env wrapper：
            对于用户自定义的`MyWrapper`，需要完成以下几步（与使用 `gym.Wrapper` 完全一致）：
            1. `MyWrapper` 继承 `gym.Wrapper`，依据需求实现其中的 `step`, `reset` 等函数
            2. 使用 `env = MyWrapper(env)` 来得到新的经过包装的环境
    - model
        对于policy中使用的model，我们对其也实现了和 `gym.Wrapper` 相似的封装，以实现对 model 类更为快速方便的更改。
        - 使用：
            已经定义好的wrapper统一放在 `nervex.armor.model_wrappers.py` 下以方便查看。对于使用 wrapper，可以按照如下规则：
            ```
            model = wrapper(model, wrapper_name='your_wrapper_name', **kwargs)
            ```
            * wrapper可以你所需要使用的任何wrapper
                * 自定义wrapper使用参考下一节
            * wrapper_name为已经注册的任意wrapper的名称。如果是自定义的wrapper，注册的时候需要提供名称。
            * kwargs部分为该wrapper所需要的参数
            * 在此情况下，得到的 `model` 可以像原本的 model 那样去使用。例如，当调用 model.forward 的时候，会优先调用 wrapper 中定义的 forward 函数。如果没有定义的话，会到下一层的 wrapper 中继续寻找。
        - 定义自己的 model wrapper：
            对于用户自定义的`MyWrapper`，需要完成以下几步：
            1. 继承 `nervex.armor.model_wrappers.IModelWrapper`，该类是 model 所使用的 wrapper 的基类。
            2. 在 `MyWrapper` 中，依据需求实现所需要的 forward 等函数。
            3. 将 `MyWrapper` 通过 `register_wrapper()` 的方法添加到 `armor_wrapper.wrapper_name_map` 这个字典中。如此一来，便可以通过 `add_wrapper` 方便地对 armor 进行添加 wrapper 的操作。
        - 调用流程：
            .. image:: wrapper_structure.png

            .. image:: wrapper_call.png
    - learner
        model中用到wrapper的地方比较少，主要表现为计时相关的time wrapper。


Hook
--------------------
概述：
    Hook，钩子，可以通过在钩子内使得外部函数在被调用的时候，自动调用钩子内定义好的函数。在程序中，对于一段封装得较好的代码，如果需要修改的话，也许要花费相当的精力。Hook函数就是由此被创造出来的。代码作者可以在一段代码中的任意位置暴露出钩子，而用户可以在钩子中实现自己所需要的功能，这样当代码运行到指定位置的时候，钩子会被触发，钩子中定义好的函数会被自动调用，从而实现快速修改代码的功能。
用处：
    nervex中使用 hook 主要是在 learner 中。
    - learner
        在nervex中，learner 的训练部分可以简化如下：
        ```
        # before_run
        for i in range(max_iter):
            # before_iter
            self._policy.forward(data)
            # after_iter
        # after_run
        ```
        而learner里面用到的hook定义了四个位置，分别为['before_run', 'after_run', 'before_iter', 'after_iter']。如上伪代码注释所示。当程序运行到指定位置的时候，该hook上所注册的函数将会被调用。
        - 使用：
            nervex已经实现了许多常用的hook，并提供了方便简单的调用方法。可以通过简单的cfg去调用hook，如下：
            ```
            hook_cfg = dict(
                save_ckpt_after_iter=20, # 在 after_iter 位置添加了名称为 save_ckpt 的 hook，每隔20个iter会存一次ckpt
                save_ckpt_after_run=True, # 在 after_run 位置添加了名称为 save_ckpt 的 hook，训练完毕的时候会存一次ckpt
            ) 
            ```
            至此，nervex在初始化learner的时候会自动根据cfg的内容进行hook注册，以保证相关功能能够正常进行。
        - 定义自己的 hook：
            对于用户自定义的 `MyHook`，需要完成以下几步：
            1. 继承 `nervex.worker.learner.learner_hook.LearnerHook`。该类是所有 learner 中使用的 hook 的基类。
            2. 在 `MyHook` 中实现 `__call__` 方法。`__call__` 方法的输入是一个 learner 的实例。通过该实例，hook可以对learner中的任意变量进行操作。
            3. 调用 `register_learner_hook()` 对自定义的 `MyWrapper` 进行注册，需要提供wrapper名称。
            4. 现在已经可以在cfg中使用自定义的 `MyWrapper`了。
        - 调用流程：
            .. image:: hook_call.png




