Cartpole
~~~~~~~~~~~~~~~~~~

Overview
==========
The inverted pendulum problem is a classic control problem in reinforcement learning. Cartpole is a discrete control task in the inverted pendulum problem. In the game there is a car with a pole on it. The cart slides side to side on a smooth and frictionless track to keep the pole upright. As shown below.

.. image:: ./images/cartpole.gif
   :align: center

Install
========

Installation Method
---------------------

The Cartpole environment is built into the gym, and you can install the gym directly. Its environment id is \ ``CartPole-v0``\.

.. code:: shell

    pip install gym
    
Verify Installation
--------------------

Run the following command on the Python command line to verify that the installation is successful.

.. code:: shell

    import gym
    env = gym.make('CartPole-v0')
    obs = env.reset()
    print(obs)

Environment Introduction
===========================

Action Space
------------

The action space of Cartpole belongs to the discrete action space, and there are two discrete actions, namely left shift and right shift.

- \ ``Left Move`` \: 0 means to move the agent to the left.

- \ ``Right move`` \: 1 means to move the agent to the right.

Using the gym environment space definition can be expressed as:

.. code:: python
    
    action_space = spaces.Discrete(2)

State Space
------------

Cartpole's state space has 4 elements, which are:

- \ ``Cart Position`` \: Cart position, in the range \ ``[-4.8, 4.8]`` \.
  
- \ ``Cart Velocity`` \: The speed of the cart, in the range \ ``[-inf, inf]`` \.

- \ ``Pole Angle`` \: The angle of the pole, in the range \ ``[-24 deg, 24 deg]``\.

- \ ``Pole Angular Velocity`` \: The angular velocity of the pole, in the range \ ``[-inf, inf]``\.


Reward Space
-------------
Each step will receive a reward of 1 until the episode terminates (the termination state will also receive a reward of 1).


Termination Condition
-----------------------
The termination condition for each episode of the Cartpole environment is any of the following:

- The angle of the rod offset is more than 12 degrees.
  
- The cart is out of bounds, and the distance is usually set as 2.4.
  
- Reaching the maximum step of episode, whose default is 200.
  

When Does the Cartpole Mission Count as a Victory
---------------------------------------------------

When the average episode reward for 100 trials reaches 195 or more, the game is considered a victory.


Others
========

Store Video
---------------

Some environments have their own rendering plug-ins, but DI-engine does not support the rendering plug-ins that come with the environment, but generates video recordings by saving the logs during training. For details, please refer to the Visualization & Logging section under the DI-engine `official documentation <https://opendilab.github.io/DI-engine/quick_start/index.html>`__ Quick start chapter.

DI-zoo Runnable Code Example
==============================

The following provides a complete cartpole environment config, using the DQN algorithm as the policy. Please run the \ ``cartpole_dqn_main.py`` \ file in the \ ``DI-engine/dizoo/classic_control/cartpole/entry`` \ directory, as follows.

.. code:: python

    import os
    import gym
    from tensorboardX import SummaryWriter

    from ding.config import compile_config
    from ding.worker import BaseLearner, SampleSerialCollector, InteractionSerialEvaluator, AdvancedReplayBuffer
    from ding.envs import BaseEnvManager, DingEnvWrapper
    from ding.policy import DQNPolicy
    from ding.model import DQN
    from ding.utils import set_pkg_seed
    from ding.rl_utils import get_epsilon_greedy_fn
    from dizoo.classic_control.cartpole.config.cartpole_dqn_config import cartpole_dqn_config


    # Get DI-engine form env class
    def wrapped_cartpole_env():
        return DingEnvWrapper(gym.make('CartPole-v0'))


    def main(cfg, seed=0):
        cfg = compile_config(
            cfg,
            BaseEnvManager,
            DQNPolicy,
            BaseLearner,
            SampleSerialCollector,
            InteractionSerialEvaluator,
            AdvancedReplayBuffer,
            save_cfg=True
        )
        collector_env_num, evaluator_env_num = cfg.env.collector_env_num, cfg.env.evaluator_env_num
        collector_env = BaseEnvManager(env_fn=[wrapped_cartpole_env for _ in range(collector_env_num)], cfg=cfg.env.manager)
        evaluator_env = BaseEnvManager(env_fn=[wrapped_cartpole_env for _ in range(evaluator_env_num)], cfg=cfg.env.manager)

        # Set random seed for all package and instance
        collector_env.seed(seed)
        evaluator_env.seed(seed, dynamic_seed=False)
        set_pkg_seed(seed, use_cuda=cfg.policy.cuda)

        # Set up RL Policy
        model = DQN(**cfg.policy.model)
        policy = DQNPolicy(cfg.policy, model=model)

        # Set up collection, training and evaluation utilities
        tb_logger = SummaryWriter(os.path.join('./{}/log/'.format(cfg.exp_name), 'serial'))
        learner = BaseLearner(cfg.policy.learn.learner, policy.learn_mode, tb_logger, exp_name=cfg.exp_name)
        collector = SampleSerialCollector(
            cfg.policy.collect.collector, collector_env, policy.collect_mode, tb_logger, exp_name=cfg.exp_name
        )
        evaluator = InteractionSerialEvaluator(
            cfg.policy.eval.evaluator, evaluator_env, policy.eval_mode, tb_logger, exp_name=cfg.exp_name
        )
        replay_buffer = AdvancedReplayBuffer(cfg.policy.other.replay_buffer, tb_logger, exp_name=cfg.exp_name)

        # Set up other modules, etc. epsilon greedy
        eps_cfg = cfg.policy.other.eps
        epsilon_greedy = get_epsilon_greedy_fn(eps_cfg.start, eps_cfg.end, eps_cfg.decay, eps_cfg.type)

        # Training & Evaluation loop
        while True:
            # Evaluating at the beginning and with specific frequency
            if evaluator.should_eval(learner.train_iter):
                stop, reward = evaluator.eval(learner.save_checkpoint, learner.train_iter, collector.envstep)
                if stop:
                    break
            # Update other modules
            eps = epsilon_greedy(collector.envstep)
            # Sampling data from environments
            new_data = collector.collect(train_iter=learner.train_iter, policy_kwargs={'eps': eps})
            replay_buffer.push(new_data, cur_collector_envstep=collector.envstep)
            # Training
            for i in range(cfg.policy.learn.update_per_collect):
                train_data = replay_buffer.sample(learner.policy.get_attribute('batch_size'), learner.train_iter)
                if train_data is None:
                    break
                learner.train(train_data, collector.envstep)
        # evaluate
        evaluator_env = BaseEnvManager(env_fn=[wrapped_cartpole_env for _ in range(evaluator_env_num)], cfg=cfg.env.manager)
        evaluator_env.enable_save_replay(cfg.env.replay_path) # switch save replay interface
        evaluator = InteractionSerialEvaluator(
            cfg.policy.eval.evaluator, evaluator_env, policy.eval_mode, tb_logger, exp_name=cfg.exp_name
        )
        evaluator.eval(learner.save_checkpoint, learner.train_iter, collector.envstep)


    if __name__ == "__main__":
        main(cartpole_dqn_config)

Experimental Results
=========================
The experimental results using the DQN algorithm are as follows. The abscissa is \ ``episode`` \, and the ordinate is \ ``reward_mean`` \.

.. image:: ./images/cartpole_dqn.png
   :align: center
   :scale: 20%

References
======================
- Cartpole `source code <https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py>`__
