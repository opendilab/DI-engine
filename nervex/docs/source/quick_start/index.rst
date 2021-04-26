Quick Start
===============================

.. toctree::
   :maxdepth: 3

Here we show how to easily deploy a Reinforcement Learning training and evaluating 
experiment on a simple CartPole environment.

NerveX provides config-wise and code-wise specifications to build RL experiments. 
Both are commonly used by existing RL platforms. In this section we use the code-level
entry to clarify the training procedure and defined modules, with the hyperparameters
for training details and NN models pre-defined in a config file. 


Config and entry
------------------

NerveX recommends using a config dictionary defined in a python file as input.

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

Each namespace belongs to a certain module in NerveX. The module can be specialized defined
by users or just use our pre-defined modules.

Set up Environments
---------------------

NerveX redefines RL environment interfaces derived from the widely used OpenAI Gym. 
For junior users, an environment wrapper is provided to simply wrap the gym env into NerveX form env.
For advanced users, it is suggested to check our Environment doc for details

The Env manager is used to manage multiple environments, single-process serially and multi-process 
parallelly. The interfaces of env manager are similar to a single env.

.. code-block:: python

    from nervex.envs import BaseEnvManager, NervexEnvWrapper

    def wrapped_cartpole_env():
        return NervexEnvWrapper(gym.make('CartPole-v0'))

    collector_env_num, evaluator_env_num = cfg.env.env_kwargs.collector_env_num, cfg.env.env_kwargs.evaluator_env_num
    collector_env = BaseEnvManager(env_fn=[wrapped_cartpole_env for _ in range(collector_env_num)])
    evaluator_env = BaseEnvManager(env_fn=[wrapped_cartpole_env for _ in range(evaluator_env_num)])

Set up Policy and NN model
----------------------------

NerveX supports most of the common policies used in RL training. The user only needs to build a PyTorch
network structure and pass it into the policy. NerveX also provides default networks to simply apply to
the environment.

For example, a DQN policy and PPO policy for CartPole can be defined as follow.

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


Set up Training modules
--------------------------




Train and evaluate the policy
---------------------------------




