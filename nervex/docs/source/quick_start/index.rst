Quick Start
===============================

.. toctree::
   :maxdepth: 3

Here we show how to easily deploy a Reinforcement Learning experiment on a simple `CartPole` 
environment using nerveX.

NerveX provides config-wise and code-wise specifications to build RL experiments. 
Both are commonly used by existing RL platforms. In this section we use the code-level
entry to clarify the training procedure and defined modules, with the hyperparameters
for training details and NN models pre-defined in a config file. 


Config and entry
------------------

NerveX recommends using a config ``dict`` defined in a python file as input.

.. code-block:: python

    cartpole_dqn_default_config = dict(
        env=dict(
            ...
        ),
        policy=dict(
            ...
        ),
        replay_buffer=dict(
            ...
        ),
        collector=dict(
            ...
        ),
        evaluator=dict(
            ...
        ),
        learner=dict(
            ...
        ),
    )

Each namespace belongs to a certain module in nerveX. The module can be specialized defined
by users or just use our pre-defined modules.

Set up Environments
---------------------

NerveX redefines RL environment interfaces derived from the widely used `OpenAI Gym <https://github.com/openai/gym>`_. 
For junior users, an environment wrapper is provided to simply wrap the gym env into NerveX form env.
For advanced users, it is suggested to check our Environment doc for details

The :class:`Env Manager <nervex.envs.BaseEnvManager>` is used to manage multiple environments, single-process serially 
or multi-process parallelly. The interfaces of `env manager` are similar to those of a simple gym env.

.. code-block:: python

    from nervex.envs import BaseEnvManager, NervexEnvWrapper

    def wrapped_cartpole_env():
        return NervexEnvWrapper(gym.make('CartPole-v0'))

    collector_env_num, evaluator_env_num = cfg.env.env_kwargs.collector_env_num, cfg.env.env_kwargs.evaluator_env_num
    collector_env = BaseEnvManager(env_fn=[wrapped_cartpole_env for _ in range(collector_env_num)])
    evaluator_env = BaseEnvManager(env_fn=[wrapped_cartpole_env for _ in range(evaluator_env_num)])

Set up Policy and NN model
----------------------------

NerveX supports most of the common policies used in RL training. Each is defined as a :class:`Policy <nervex.policy.CommonPolicy>`
class. The details of optimiaztion algorithm, data pre-processing and post-processing, control of multiple networks 
are encapsulated inside. Users only need to build a PyTorch network structure and pass into the policy. 
NerveX also provides default networks to simply apply to the environment.

For example, a `DQN` policy and `PPO` policy for CartPole can be defined as follow.

.. code-block:: python

    from nervex.policy import DQNPolicy
    from nervex.model import FCDiscreteNet

    model = FCDiscreteNet(**cfg.policy.model)
    policy = DQNPolicy(cfg.policy, model=model)

.. code-block:: python

    from nervex.policy import PPOPolicy
    from nervex.model import FCValueAC

    model = FCValueAC(**cfg.policy.model)
    policy = PPOPolicy(cfg.policy, model=model)


Set up runtime modules
--------------------------

NerveX needs to build some runtime components to manage an RL training procedure. 
A :class:`Collector <nervex.worker.BaseSerialCollector>` is used to sample and provide data for training.
A :class:`Learner <nervex.worker.BaseLearner>` is used to receive training data and conduct 
the training (including updating networks, strategy and experience pool, etc.).
An :class:`Evaluator <nervex.worker.BaseSerialEvaluator>` is build to perform the evaluation when needed.
And other components like :class:`Replay Buffer <nervex.worker.replay_buffer.IBuffer>` may be required for the
training process. All these module can be customized by config or rewritten by the user.

An example of setting up all the above is showed as follow.

.. code-block:: python

    from tensorboardX import SummaryWriter    
    from nervex.worker import BaseLearner, BaseSerialCollector, BaseSerialEvaluator
    from nervex.data import BufferManager

    tb_logger = SummaryWriter(os.path.join('./log/', 'serial'))
    learner = BaseLearner(cfg.learner, policy.learn_mode, tb_logger)
    collector = BaseSerialCollector(cfg.collector, collector_env, policy.collect_mode, tb_logger)
    evaluator = BaseSerialEvaluator(cfg.evaluator, evaluator_env, policy.eval_mode, tb_logger)
    replay_buffer = BufferManager(cfg.replay_buffer, tb_logger)

Train and evaluate the policy
---------------------------------

The training loop in nerveX can be customized arbitrarily. Usually the training process may consist of
collecting data, updating policy, updating related modules and evaluation.

Here we provide examples of off-policy training (`DQN`) and on-policy training (`PPO`) for a `CartPole`
environment.

.. code-block:: python

    from nervex.rl_utils import get_epsilon_greedy_fn
    
    # DQN training loop
    epsilon_greedy = get_epsilon_greedy_fn(eps_cfg.start, eps_cfg.end, eps_cfg.decay, eps_cfg.type)
    while True:
        if evaluator.should_eval(learner.train_iter):
            stop, reward = evaluator.eval(learner.save_checkpoint, learner.train_iter, collector.envstep)
            if stop:
                break

        eps = epsilon_greedy(learner.train_iter)
        new_data = collector.collect_data(learner.train_iter, policy_kwargs={'eps': eps})
        replay_buffer.push(new_data, cur_collector_envstep=collector.envstep)
        for i in range(cfg.policy.learn.train_iteration):
            train_data = replay_buffer.sample(learner.policy.get_attribute('batch_size'), learner.train_iter)
            if train_data is not None:
                learner.train(train_data, collector.envstep)

.. code-block:: python

    # PPO training loop
    while True:
        if evaluator.should_eval(learner.train_iter):
            stop, reward = evaluator.eval(learner.save_checkpoint, learner.train_iter, collector.envstep)
            if stop:
                break
        new_data = collector.collect_data(learner.train_iter)
        replay_buffer.push(new_data, cur_collector_envstep=collector.envstep)
        for i in range(cfg.policy.learn.train_iteration):
            train_data = replay_buffer.sample(learner.policy.get_attribute('batch_size'), learner.train_iter)
            if train_data is not None:
                learner.train(train_data, collector.envstep)
                replay_buffer.update(learner.priority_info)


Advanced features
------------------

Some advanced features in RL training which well supported by nerveX are listed below.

Epsilon Greedy & Replay start and priority
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An easy way of deploying epsilon greedy exploration when sampling data has already been shown above. It is
called by the `epsilon_greedy` function each step.

.. code-block:: python

    from nervex.rl_utils import get_epsilon_greedy_fn
    
    epsilon_greedy = get_epsilon_greedy_fn(eps_cfg.start, eps_cfg.end, eps_cfg.decay, eps_cfg.type)
    while True:
        eps = epsilon_greedy(learner.train_iter)
        ...

Initially collecting an amount of data is supported in the following way. 


.. code-block:: python

    if replay_buffer.replay_start_size() > 0:
        eps = epsilon_greedy(learner.train_iter)
        new_data = collector.collect_data(learner.train_iter, n_sample=replay_buffer.replay_start_size(), policy_kwargs={'eps': eps})
        replay_buffer.push(new_data, cur_collector_envstep=0)


The priority mechanism is widely used in RL training. Nervex adds easy-used interface to apply priority to replay
buffer, shown as follow.

.. code-block:: python

    if use_priority:
        replay_buffer.update(learner.priority_info)

Visualization & Logging
~~~~~~~~~~~~~~~~~~~~~~~~~

Some environments have a renderd surface or visualization. NerveX adds a switch to save these replays.

.. code-block:: python

    if cfg.env.env_kwargs.get('replay_path', None):
        evaluator_env.enable_save_replay([cfg.env.env_kwargs.replay_path for _ in range(evaluator_env_num)])

Similar with other Deep Learning platforms, nerveX uses tensorboard to record key parameters and results during
training. In addition to the default logging parameters, users can add their own logging parameters as follow.

.. code-block:: python

    tb_logger.add_scalar('epsilon_greedy', eps, learner.train_iter)
