from typing import List
from ding.utils import POLICY_REGISTRY
from .ddpg import DDPGPolicy


@POLICY_REGISTRY.register('td3')
class TD3Policy(DDPGPolicy):
    """
    Overview:
        Policy class of TD3 algorithm. Since DDPG and TD3 share many common things, we can easily derive this TD3 \
        class from DDPG class by changing ``_actor_update_freq``, ``_twin_critic`` and noise in model wrapper.
        Paper link: https://arxiv.org/pdf/1802.09477.pdf

    Config:

    == ====================  ========    ==================  =================================   =======================
    ID Symbol                Type        Default Value       Description                         Other(Shape)
    == ====================  ========    ==================  =================================   =======================
    1  | ``type``            str         td3                 | RL policy register name, refer    | this arg is optional,
       |                                                     | to registry ``POLICY_REGISTRY``   | a placeholder
    2  | ``cuda``            bool        False               | Whether to use cuda for network   |
    3  | ``random_``         int         25000               | Number of randomly collected      | Default to 25000 for
       | ``collect_size``                                    | training samples in replay        | DDPG/TD3, 10000 for
       |                                                     | buffer when training starts.      | sac.
    4  | ``model.twin_``     bool        True                | Whether to use two critic         | Default True for TD3,
       | ``critic``                                          | networks or only one.             | Clipped Double
       |                                                     |                                   | Q-learning method in
       |                                                     |                                   | TD3 paper.
    5  | ``learn.learning``  float       1e-3                | Learning rate for actor           |
       | ``_rate_actor``                                     | network(aka. policy).             |
    6  | ``learn.learning``  float       1e-3                | Learning rates for critic         |
       | ``_rate_critic``                                    | network (aka. Q-network).         |
    7  | ``learn.actor_``    int         2                   | When critic network updates       | Default 2 for TD3, 1
       | ``update_freq``                                     | once, how many times will actor   | for DDPG. Delayed
       |                                                     | network update.                   | Policy Updates method
       |                                                     |                                   | in TD3 paper.
    8  | ``learn.noise``     bool        True                | Whether to add noise on target    | Default True for TD3,
       |                                                     | network's action.                 | False for DDPG.
       |                                                     |                                   | Target Policy Smoo-
       |                                                     |                                   | thing Regularization
       |                                                     |                                   | in TD3 paper.
    9  | ``learn.noise_``    dict        | dict(min=-0.5,    | Limit for range of target         |
       | ``range``                       |      max=0.5,)    | policy smoothing noise,           |
       |                                 |                   | aka. noise_clip.                  |
    10 | ``learn.-``         bool        False               | Determine whether to ignore       | Use ignore_done only
       | ``ignore_done``                                     | done flag.                        | in halfcheetah env.
    11 | ``learn.-``         float       0.005               | Used for soft update of the       | aka. Interpolation
       | ``target_theta``                                    | target network.                   | factor in polyak aver
       |                                                     |                                   | -aging for target
       |                                                     |                                   | networks.
    12 | ``collect.-``       float       0.1                 | Used for add noise during co-     | Sample noise from dis
       | ``noise_sigma``                                     | llection, through controlling     | -tribution, Ornstein-
       |                                                     | the sigma of distribution         | Uhlenbeck process in
       |                                                     |                                   | DDPG paper, Gaussian
       |                                                     |                                   | process in ours.
    == ====================  ========    ==================  =================================   =======================
   """

    # You can refer to DDPG's default config for more details.
    config = dict(
        # (str) RL policy register name (refer to function "POLICY_REGISTRY").
        type='td3',
        # (bool) Whether to use cuda for network.
        cuda=False,
        # (bool) on_policy: Determine whether on-policy or off-policy. Default False in TD3.
        on_policy=False,
        # (bool) Whether use priority(priority sample, IS weight, update priority)
        # Default False in TD3.
        priority=False,
        # (bool) Whether use Importance Sampling Weight to correct biased update. If True, priority must be True.
        priority_IS_weight=False,
        # (int) Number of training samples(randomly collected) in replay buffer when training starts.
        # Default 25000 in DDPG/TD3.
        random_collect_size=25000,
        # (bool) Whether to need policy data in process transition.
        transition_with_policy_data=False,
        # (str) Action space type
        action_space='continuous',  # ['continuous', 'hybrid']
        # (bool) Whether use batch normalization for reward
        reward_batch_norm=False,
        # (bool) Whether to enable multi-agent training setting
        multi_agent=False,
        model=dict(
            # (bool) Whether to use two critic networks or only one.
            # Clipped Double Q-Learning for Actor-Critic in original TD3 paper(https://arxiv.org/pdf/1802.09477.pdf).
            # Default True for TD3, False for DDPG.
            twin_critic=True,
        ),
        # learn_mode config
        learn=dict(
            # (int) How many updates(iterations) to train after collector's one collection.
            # Bigger "update_per_collect" means bigger off-policy.
            # collect data -> update policy-> collect data -> ...
            update_per_collect=1,
            # (int) Minibatch size for gradient descent.
            batch_size=256,
            # (float) Learning rates for actor network(aka. policy).
            learning_rate_actor=1e-3,
            # (float) Learning rates for critic network(aka. Q-network).
            learning_rate_critic=1e-3,
            # (bool) Whether ignore done(usually for max step termination env. e.g. pendulum)
            # Note: Gym wraps the MuJoCo envs by default with TimeLimit environment wrappers.
            # These limit HalfCheetah, and several other MuJoCo envs, to max length of 1000.
            # However, interaction with HalfCheetah always gets done with False,
            # Since we inplace done==True with done==False to keep
            # TD-error accurate computation(``gamma * (1 - done) * next_v + reward``),
            # when the episode step is greater than max episode step.
            ignore_done=False,
            # (float) target_theta: Used for soft update of the target network,
            # aka. Interpolation factor in polyak averaging for target networks.
            # Default to 0.005.
            target_theta=0.005,
            # (float) discount factor for the discounted sum of rewards, aka. gamma.
            discount_factor=0.99,
            # (int) When critic network updates once, how many times will actor network update.
            # Delayed Policy Updates in original TD3 paper(https://arxiv.org/pdf/1802.09477.pdf).
            # Default 1 for DDPG, 2 for TD3.
            actor_update_freq=2,
            # (bool) Whether to add noise on target network's action.
            # Target Policy Smoothing Regularization in original TD3 paper(https://arxiv.org/pdf/1802.09477.pdf).
            # Default True for TD3, False for DDPG.
            noise=True,
            # (float) Sigma for smoothing noise added to target policy.
            noise_sigma=0.2,
            # (dict) Limit for range of target policy smoothing noise, aka. noise_clip.
            noise_range=dict(
                # (int) min value of noise
                min=-0.5,
                # (int) max value of noise
                max=0.5,
            ),
        ),
        # collect_mode config
        collect=dict(
            # (int) How many training samples collected in one collection procedure.
            # Only one of [n_sample, n_episode] shoule be set.
            # n_sample=1,
            # (int) Cut trajectories into pieces with length "unroll_len".
            unroll_len=1,
            # (float) It is a must to add noise during collection. So here omits "noise" and only set "noise_sigma".
            noise_sigma=0.1,
        ),
        eval=dict(),  # for compability
        other=dict(
            replay_buffer=dict(
                # (int) Maximum size of replay buffer. Usually, larger buffer size is better.
                replay_buffer_size=100000,
            ),
        ),
    )

    def _monitor_vars_learn(self) -> List[str]:
        """
        Overview:
            Return the necessary keys for logging the return dict of ``self._forward_learn``. The logger module, such \
            as text logger, tensorboard logger, will use these keys to save the corresponding data.
        Returns:
            - necessary_keys (:obj:`List[str]`): The list of the necessary keys to be logged.
        """
        return ["q_value", "loss", "lr", "entropy", "target_q_value", "td_error"]
