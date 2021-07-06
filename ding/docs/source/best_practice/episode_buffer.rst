How to use Episode Replay Buffer?
===================================

Guideline
^^^^^^^^^^^^^^
In some algorithms or envs, collecting and storing a whole episode is more useful than separated samples. For example: In chess, go or card games, players get reward only when the game is over; Some algorithms, for example, `Hindsight Experience Replay(HER) <https://arxiv.org/abs/1707.01495>`_, must sample out a whole episode and operate on it. Therefore, DI-engine implements ``EpisodeReplayBuffer`` (``ding/worker/buffer/episode_buffer.py``), where each element is no longer a training sample, but an episode.

In this section, we will introduce how to use such episode buffer in a training pipeline. We will take algorithm `HER` as an example. The source code is at ``dizoo/classic_control/bitflip/entry/bitflip_dqn_main.py``


.. code:: python

    # Other components' initialization are the same as normal pipeline.
    # User only need to change the buffer type to use episode buffer.
    # PS. Episode buffer's config is similar to naive buffer.
    replay_buffer = EpisodeReplayBuffer(cfg.policy.other.replay_buffer, 'episode')

    # Training & Evaluation loop
    while True:
        # Evaluate
        # ...

        # Sample new episodes from environments, then push into episode buffer.
        new_episode = collector.collect(train_iter=learner.train_iter, policy_kwargs={'eps': eps})
        replay_buffer.push(new_episode, cur_collector_envstep=collector.envstep)
        
        # Train
        for i in range(cfg.policy.learn.update_per_collect):
            sample_size = learner.policy.get_attribute('batch_size')
            train_episode = replay_buffer.sample(sample_size, learner.train_iter)
            if train_episode is not None:
                train_data = []
                # Get train samples from each episode, and use all samples to train.
                for e in train_episode:
                    train_data.extend(policy.collect_mode.get_train_sample(e))
                learner.train(train_data, collector.envstep)
    
There is one thing you should pay attention to. Because episode lengths are **different** and may have **high variance**, if you collect, store and sample complete episodes, it may be hard to tell how many training samples you generate and use. As a result, it is more common in sampling process to regulate that **how many episodes to sample**, and **how many training samples to get from one episode**. The **product** of these two is the actual batch size.

Therefore, the actual code of `Her` is as follows:

.. code:: python

    replay_buffer = EpisodeReplayBuffer(cfg.policy.other.replay_buffer, 'episode')
    # Initialize her_model, which is used to pre-process episode
    her_cfg = cfg.policy.other.get('her', None)
    if her_cfg is not None:
        her_model = HerRewardModel(her_cfg, cfg.policy.cuda)

    while True:
        new_episode = collector.collect(train_iter=learner.train_iter, policy_kwargs={'eps': eps})
        replay_buffer.push(new_episode, cur_collector_envstep=collector.envstep)
        for i in range(cfg.policy.learn.update_per_collect):
            # Set a new key `episode_size`; Do not use common `batch_size`
            if her_cfg and her_model.episode_size is not None:
                sample_size = her_model.episode_size
            else:
                sample_size = learner.policy.get_attribute('batch_size')
            train_episode = replay_buffer.sample(sample_size, learner.train_iter)
            if train_episode is not None:
                train_data = []
                # First pre-process episodes with her model
                # Processed episode is no longer a complete episode, but a list of fixed number transitions.
                if her_cfg is not None:
                    her_episodes = []
                    for e in train_episode:
                        her_episodes.extend(her_model.estimate(e))
                    train_episode.extend(her_episodes)
                for e in train_episode:
                    train_data.extend(policy.collect_mode.get_train_sample(e))
                learner.train(train_data, collector.envstep)
