Quick Start
===============================

.. toctree::
   :maxdepth: 3


.. image:: 
   images/cartpole_cmp.gif
   :align: center

As a kickoff, we will illustrate how to launch an RL experiment on a simple ``CartPole`` environment (as shown in the above figure) using DI-engine.


Concretely, we will define a training pipeline in a single python file that specifies the training hyper-parameters, sampling and evaluation environments, the neural networks for the RL agents, as well as the training workflow.

Instantiate the run-time config
-------------------------------

The first step to build a training workflow is to specify the training configuration.
DI-engine prefers a nested python dict object to represent all parameters and configurations of an RL experiment, for example:

.. code-block:: python

    cartpole_dqn_config = dict(
        exp_name="cartpole_dqn",
        env=dict(
            collector_env_num=8,
            evaluator_env_num=5,
        ),
        policy=dict(
            model=dict(
                encoder_hidden_size_list=[128, 128, 64],
            ),
            discount_factor=0.97,
        ),
        ......
    )

.. note ::

    For the specific example, you can refer to:

      - config:  ``dizoo/classic_control/cartpole/config/cartpole_dqn_config.py``
      - main: ``dizoo/classic_control/cartpole/entry/cartpole_dqn_main.py``

    and you just need to run this experiment following the next command:

    .. code:: bash

        python3 -u dizoo/classic_control/cartpole/entry/cartpole_dqn_main.py

DI-engine provides default configs for all modules, and also a helper function ``compile_config`` to merge the default configs of these modules into a run-time config object (a ``EasyDict`` object that can be accessed by string key ``cfg["env"]`` or attribute ``cfg.env``):

.. code-block:: python

    from ding.config import compile_config
    from ding.envs import BaseEnvManager, DingEnvWrapper
    from ding.model import DQN
    from ding.policy import DQNPolicy
    from ding.worker import BaseLearner, SampleCollector, BaseSerialEvaluator, AdvancedReplayBuffer
    from dizoo.classic_control.cartpole.config.cartpole_dqn_config import cartpole_dqn_config

    # compile config
    cfg = compile_config(
        cartpole_dqn_config,
        BaseEnvManager,
        DQNPolicy,
        BaseLearner,
        SampleCollector,
        BaseSerialEvaluator,
        AdvancedReplayBuffer,
        save_cfg=True
    )

In this example, we only present the procedure to specify config in the entry file. In the following section, we construct the RL pipeline in the same entry file based on the specified config.

Please note that DI-engine also supports running an RL experiment directly according to a given config file, e.g. 

.. code:: bash

    ding -m serial -c cartpole_dqn_config.py -s 0

For more design details, please refer to the `Config <../key_concept/index.html#config>`_ section and `Entry <../key_concept/index.html#entry>`_.


Initialize the Environments
---------------------------

The RL agents interact with the environment to collect training data or test its performance.
DI-engine provides enhanced RL environment interfaces derived from the widely used `OpenAI Gym <https://github.com/openai/gym>`_. 
You can simply wrap the gym environment into the DI-engine environment by using the environment wrapper :class:`DingEnvWrapper <ding.env.DingEnvWrapper>`.
You can also construct a more complex environment class following the guidelines in `Environment <../key_concept/index.html#env>`_ section.

The :class:`Env Manager <ding.envs.BaseEnvManager>` is used to manage multiple vectorized environments, usually implemented by
multi-processes parallelly. The interfaces of `env manager` are similar to those of a simple gym env. Here we show a case
of using :class:`BaseEnvManager <ding.envs.BaseEnvManager>` to build environments for collection and evaluation.

.. code-block:: python

    import gym

    def wrapped_cartpole_env():
        return DingEnvWrapper(gym.make('CartPole-v0'))

    collector_env_num, evaluator_env_num = cfg.env.collector_env_num, cfg.env.evaluator_env_num
    collector_env = BaseEnvManager(env_fn=[wrapped_cartpole_env for _ in range(collector_env_num)], cfg=cfg.env.manager)
    evaluator_env = BaseEnvManager(env_fn=[wrapped_cartpole_env for _ in range(evaluator_env_num)], cfg=cfg.env.manager)

In order to ensure the reproducibility of experiement, we setup the seed of environments and common packages. 

.. code-block:: python

    from ding.utils import set_pkg_seed

    collector_env.seed(seed=0)
    evaluator_env.seed(seed=0, dynamic_seed=False)
    set_pkg_seed(seed=0, use_cuda=cfg.policy.cuda)

Set up the Policy and NN models
-------------------------------

DI-engine supports most of the common policies used in RL training. Each is defined as a :class:`Policy <ding.policy.CommonPolicy>`
class. The details of optimization algorithms, data pre-processing and post-processing, usage of neural networks
are encapsulated inside. Users only need to build a PyTorch network structure and pass it into the policy. 

DI-engine also provides default networks to simply apply to the environment. For some complex RL methods, users can imitate the interfaces of these default models and customize their own networks.

For example, a ``DQN`` policy for ``CartPole`` can be defined as follow.

.. code-block:: python

    model = DQN(**cfg.policy.model)
    policy = DQNPolicy(cfg.policy, model=model)


Define the Execution Modules
----------------------------

DI-engine needs to build some execution components to manage an RL training procedure. 
A :class:`Collector <ding.worker.collector.SampleCollector>` is used to sample and provide data for training.
A :class:`Learner <ding.worker.learner.BaseLearner>` is used to receive training data and conduct 
the training (including updating networks, strategy and etc.).
An :class:`Evaluator <ding.worker.collector.BaseSerialEvaluator>` is built to perform the evaluation when needed.
And other components like :class:`Replay Buffer <ding.worker.replay_buffer.AdvancedReplayBuffer>` may be required for the
training process. All these modules can be customized by config or rewritten by the user.

An example of setting all the above is shown as follows.

.. code-block:: python

    import os
    from tensorboardX import SummaryWriter    

    tb_logger = SummaryWriter(os.path.join('./{}/log/'.format(cfg.exp_name), 'serial'))
    learner = BaseLearner(cfg.policy.learn.learner, policy.learn_mode, tb_logger, exp_name=cfg.exp_name)
    collector = SampleCollector(
        cfg.policy.collect.collector, collector_env, policy.collect_mode, tb_logger, exp_name=cfg.exp_name
    )
    evaluator = BaseSerialEvaluator(
        cfg.policy.eval.evaluator, evaluator_env, policy.eval_mode, tb_logger, exp_name=cfg.exp_name
    )
    replay_buffer = AdvancedReplayBuffer(cfg.policy.other.replay_buffer, tb_logger, exp_name=cfg.exp_name)

Aggregate the Training and Evaluation Pipelines
-----------------------------------------------

The training loop in DI-engine can be customized arbitrarily. Usually the training process may consist of
collecting data, updating policy, updating related modules and evaluation.

Here we provide examples of off-policy training (``DQN``) for a ``CartPole`` environment. For more algorithms, you can refer to dizoo.

.. code-block:: python

    from ding.rl_utils import get_epsilon_greedy_fn
    
    # DQN training loop
    eps_cfg = cfg.policy.other.eps
    epsilon_greedy = get_epsilon_greedy_fn(eps_cfg.start, eps_cfg.end, eps_cfg.decay, eps_cfg.type)
    max_iterations = int(1e8)
    for _ in range(max_iterations):
        if evaluator.should_eval(learner.train_iter):
            stop, reward = evaluator.eval(learner.save_checkpoint, learner.train_iter, collector.envstep)
            if stop:
                break
        eps = epsilon_greedy(collector.envstep)
        new_data = collector.collect(train_iter=learner.train_iter, policy_kwargs={'eps': eps})
        replay_buffer.push(new_data, cur_collector_envstep=collector.envstep)
        for i in range(cfg.policy.learn.update_per_collect):
            train_data = replay_buffer.sample(learner.policy.get_attribute('batch_size'), learner.train_iter)
            if train_data is not None:
                learner.train(train_data, collector.envstep)

.. note::
   The users can refer to the complete demo in ``dizoo/classic_control/cartpole/entry/cartpole_dqn_main.py``.

Other Utilities
------------------

DI-engine supports various useful tools in common RL training, as shown in follows.

Epsilon Greedy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An easy way of deploying epsilon greedy exploration when sampling data is shown as follows:

.. code-block:: python

    from ding.rl_utils import get_epsilon_greedy_fn
    
    eps_cfg = cfg.policy.other.eps
    epsilon_greedy = get_epsilon_greedy_fn(eps_cfg.start, eps_cfg.end, eps_cfg.decay, eps_cfg.type)
    while True:
        eps = epsilon_greedy(learner.train_iter)
        ...

Firstly, you should call ``get_epsilon_greedy_fn`` to acquire an eps-greedy function. Then, you should call ``epsilon_greedy`` function at each step. The epsilon decay strategy can be configured by you, for example, start value, end value, type of decay(linear, exponential). And you can control whether it decays by env steps or train iteration.


Visualization & Logging
~~~~~~~~~~~~~~~~~~~~~~~~~

Some environments have a rendering visualization. DI-engine doesn't use render interface, but supports saving replay videos instead.
After training, users can add the code shown below to enable this function. If everything works well, you can find some videos with the ``.mp4`` suffix in directory ``replay_path`` (some GUI interfaces are normal).


.. code-block:: python

    evaluator_env = BaseEnvManager(env_fn=[wrapped_cartpole_env for _ in range(evaluator_env_num)], cfg=cfg.env.manager)
    cfg.env.replay_path = './video'  # indicate save replay directory path
    evaluator_env.seed(seed=0, dynamic_seed=False)
    evaluator_env.enable_save_replay(cfg.env.replay_path)  # switch save replay interface
    evaluator = BaseSerialEvaluator(
        cfg.policy.eval.evaluator, evaluator_env, policy.eval_mode, tb_logger, exp_name=cfg.exp_name
    )
    evaluator.eval(learner.save_checkpoint, learner.train_iter, collector.envstep)

.. note::

  If users want to visualize with a trained policy, please refer to ``dizoo/classic_control/cartpole/entry/cartpole_dqn_eval.py`` to construct a user-defined evaluation function, and indicate two fields ``env.replay_path`` and ``policy.learn.learner.hook.load_ckpt_before_run`` in config. An example is shown as follows:

  .. code-block:: python
  
    config = dict(
        env=dict(
            replay_path='your_replay_save_dir_path',
        ),
        policy=dict(
            ...,
            load_path='your_ckpt_path',
            ...,
        ),
    )

.. tip::

    All new RL environments can define their own ``enable_save_replay`` method to specify how to generate replay files. DI-engine utilizes ``gym wrapper (coupled with ffmpeg)`` to generate replays for some traditional environments. If users encounter some errors in recording videos by ``gym wrapper``, you should install ``ffmpeg`` first.


Similar with other Deep Learning platforms, DI-engine uses tensorboard to record key parameters and results during
training. In addition to the default logging parameters, users can add their own logging parameters as follows.

.. code-block:: python

    tb_logger.add_scalar('epsilon_greedy', eps, learner.train_iter)

If you want to know more details about default information recorded in tensorboard, see our 
`tensorboard and logging demo <./tb_demo.html>`_ for a
DQN experiment.

Loading & Saving checkpoints
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

It is usually needed to save and resume an experiment with model checkpoints. 
DI-engine saves and loads checkpoints in the same way as PyTorch.

.. code-block:: python

    ckpt_path = 'path/to/your/ckpt'
    state_dict = torch.load(ckpt_path, map_location='cpu')
    learner.policy.load_state_dict(state_dict)
    learner.info('{} load ckpt in {}'.format(learner.name, ckpt_path))
    
    ...

    dirname = './ckpt_{}'.format(learner.name)
    os.mkdir(dirname, exist_ok=True)
    ckpt_name = 'iteration_{}.pth.tar'.format(learner.last_iter.val)
    path = os.path.join(dirname, ckpt_name)
    state_dict = learner.policy.state_dict()
    torch.save(state_dict, path)
    learner.info('{} save ckpt in {}'.format(learner.name, path))

To deploy this in a more elegant way, DI-engine is configured to use 
:class:`Learner Hook <ding.worker.learner.learner_hook.LearnerHook>` to handle these cases. The saving hook is 
automatically called after training iterations. And to load & save checkpoints at the beginning and 
in the end, users can simply add one-line code before & after training as follows.

.. code-block:: python
    
    learner.call_hook('before_run')

    # training loop
    while True:
        ...
    
    learner.call_hook('after_run')

For more information, please take a look to `Wrapper & Hook Overview <../feature/wrapper_hook_overview.html>`_ doc.

(Note: This page is based on commit )
