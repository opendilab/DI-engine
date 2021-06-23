from nervex.utils import POLICY_REGISTRY
from .ddpg import DDPGPolicy

@POLICY_REGISTRY.register('td3')
class TD3Policy(DDPGPolicy):
    r"""
    Overview:
        Policy class of TD3 algorithm. Since DDPG and TD3 share many common things, we can easily derive this TD3
        class from DDPG class by changing ``_actor_update_freq``, ``_twin_critic`` and noise in model wrapper.
    Property:
        learn_mode, collect_mode, eval_mode

    Config:
           == ====================  ========    ==================  ===============================================     ========================================================
           ID Symbol                Type        Default Value       Description                                         Other(Shape)
           == ====================  ========    ==================  ===============================================     ========================================================
           1  ``type``              str         td3                 | RL policy register name, refer to                 | this arg is optional,
                                                                    | registry ``POLICY_REGISTRY``                      | a placeholder
           2  ``cuda``              bool        True                | Whether to use cuda for network                   |
           3  |``random_``          int         25000               | Number of training samples(randomly collected)    | Default to 25000 for DDPG/TD3, 10000 for sac.
              |``collect_size``                                     | in replay buffer when training starts.            |
           4  |``model.twin_critic``bool        True                | Whether to use two critic networks or only one.   | Default True for TD3, False for DDPG.
              |                                                     |                                                   | Clipped Double Q-learning method in TD3 paper.
           5  | ``learn.learning_`` float       1e-3                | Learning rate for actor network(aka. policy).     |
              | ``rate_actor``                                      |                                                   |
           6  | ``learn.learning_`` float       1e-3                | Learning rates for critic network                 |
              | ``rate_critic``                                     | (aka. Q-network).                                 |
           7  | ``learn.actor_``    int         2                   | When critic network updates once,                 | Default 2 for TD3, 1 for DDPG.
              | ``update_freq``                                     | how many times will actor network update.         | Delayed Policy Updates method in TD3 paper.
           8  | ``learn.noise``     bool        True                | Whether to add noise on target network's action.  | Default True for TD3, False for DDPG.
              |                                                     |                                                   | Target Policy Smoothing Regularization in TD3 paper.
           9  | ``learn.noise_``    dict        | dict(min=-0.5,    | Limit for range of target policy smoothing noise, |
              | ``range``                       |      max=0.5,)    | aka. noise_clip.                                  |
           10 | ``learn.-``         bool        False               | Determine whether to ignore done flag.            | use ignore_done only in halfcheetah env.
              | ``ignore_done``                                     |                                                   |
           11 | ``learn.-``         float       0.005               | Used for soft update of the target network.       | aka. Interpolation factor in polyak averaging
              | ``target_theta``                                    |                                                   | for target networks.
           12 | ``collect.-``       float       0.1                 | Used for add noise during collection, through     | sample noise from distribution, like Ornstein-Uhlenbeck
              | ``noise_sigma``                                     | controlling the sigma of distribution             | process in DDPG paper, Guassian process n ours.
           == ====================  ========   `==================  ===============================================     ========================================================
   """

    # You can refer to DDPG's default config for more details.
    config = dict(
        # (str) RL policy register name (refer to function "POLICY_REGISTRY").
        type='td3',
        # (bool) Whether to use cuda for network.
        cuda=False,
        # (bool type) on_policy: Determine whether on-policy or off-policy.
        # on-policy setting influences the behaviour of buffer.
        # Default False in TD3.
        on_policy=False,
        # (bool) Whether use priority(priority sample, IS weight, update priority)
        # Default False in TD3.
        priority=False,
        # (bool) Whether use Importance Sampling Weight to correct biased update. If True, priority must be True.
        priority_IS_weight=False,
        # (int) Number of training samples(randomly collected) in replay buffer when training starts.
        # Default 25000 in DDPG/TD3.
        random_collect_size=25000,
        model=dict(
            # (bool) Whether to use two critic networks or only one.
            # Clipped Double Q-Learning for Actor-Critic in original TD3 paper.
            # Default True for TD3, False for DDPG.
            twin_critic=True,
        ),
        learn=dict(
            multi_gpu=False,
            # How many updates(iterations) to train after collector's one collection.
            # Bigger "update_per_collect" means bigger off-policy.
            # collect data -> update policy-> collect data -> ...
            update_per_collect=1,
            # Minibatch size for gradient descent.
            batch_size=256,
            # Learning rates for actor network(aka. policy).
            learning_rate_actor=1e-3,
            # Learning rates and critic network(aka. Q-network).
            learning_rate_critic=1e-3,
            # (bool) Whether ignore done(usually for max step termination env. e.g. pendulum)
            # Note: Gym wraps the MuJoCo envs by default with TimeLimit environment wrappers.
            # These limit HalfCheetah, and several other MuJoCo envs, to max length of 1000.
            # However, interaction with HalfCheetah always gets done with done is False,
            # Since we inplace done==True with done==False to keep
            # TD-error accurate computation(``gamma * (1 - done) * next_v + reward``),
            # when the episode step is greater than max episode step.
            ignore_done=False,
            # (float type) target_theta: Used for soft update of the target network,
            # aka. Interpolation factor in polyak averaging for target networks.
            # Default to 0.005.
            target_theta=0.005,
            # (float) discount factor for the discounted sum of rewards, aka. gamma.
            discount_factor=0.99,
            # (int) When critic network updates once, how many times will actor network update.
            # Delayed Policy Updates in original TD3 paper.
            # Default 1 for DDPG, 2 for TD3.
            actor_update_freq=2,
            # (bool) Whether to add noise on target network's action.
            # Target Policy Smoothing Regularization in original TD3 paper.
            # Default True for TD3, False for DDPG.
            noise=True,
            # (float) Sigma for smoothing noise added to target policy.
            noise_sigma=0.2,
            # (dict) Limit for range of target policy smoothing noise, aka. noise_clip.
            noise_range=dict(
                min=-0.5,
                max=0.5,
            ),
        ),
        collect=dict(
            # n_sample=1,
            # (int) Cut trajectories into pieces with length "unroll_len".
            unroll_len=1,
            # It is a must to add noise during collection. So here omits "noise" and only set "noise_sigma".
            noise_sigma=0.1,
        ),
        other=dict(
            replay_buffer=dict(
                # (int) Maximum size of replay buffer
                replay_buffer_size=1000000,
            ),
        ),
    )
