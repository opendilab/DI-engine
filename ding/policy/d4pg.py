from typing import List, Dict, Any, Tuple, Union
import torch
import copy

from ding.torch_utils import Adam, to_device
from ding.rl_utils import get_train_sample
from ding.rl_utils import dist_nstep_td_data, dist_nstep_td_error, get_nstep_return_data
from ding.model import model_wrap
from ding.utils import POLICY_REGISTRY
from .ddpg import DDPGPolicy
from .common_utils import default_preprocess_learn
import numpy as np


@POLICY_REGISTRY.register('d4pg')
class D4PGPolicy(DDPGPolicy):
    r"""
    Overview:
        Policy class of D4PG algorithm.
    Property:
        learn_mode, collect_mode, eval_mode
    Config:
        == ====================  ========    =============  =================================   =======================
        ID Symbol                Type        Default Value  Description                         Other(Shape)
        == ====================  ========    =============  =================================   =======================
        1  ``type``              str         d4pg           | RL policy register name, refer    | this arg is optional,
                                                            | to registry ``POLICY_REGISTRY``   | a placeholder
        2  ``cuda``              bool        True           | Whether to use cuda for network   |
        3  | ``random_``         int         25000          | Number of randomly collected      | Default to 25000 for
           | ``collect_size``                               | training samples in replay        | DDPG/TD3, 10000 for
           |                                                | buffer when training starts.      | sac.
        5  | ``learn.learning``  float       1e-3           | Learning rate for actor           |
           | ``_rate_actor``                                | network(aka. policy).             |
        6  | ``learn.learning``  float       1e-3           | Learning rates for critic         |
           | ``_rate_critic``                               | network (aka. Q-network).         |
        7  | ``learn.actor_``    int         1              | When critic network updates       | Default 1
           | ``update_freq``                                | once, how many times will actor   |
           |                                                | network update.                   |
        8  | ``learn.noise``     bool        False          | Whether to add noise on target    | Default False for
           |                                                | network's action.                 | D4PG.
           |                                                |                                   | Target Policy Smoo-
           |                                                |                                   | thing Regularization
           |                                                |                                   | in TD3 paper.
        9  | ``learn.-``         bool        False          | Determine whether to ignore       | Use ignore_done only
           | ``ignore_done``                                | done flag.                        | in halfcheetah env.
        10 | ``learn.-``         float       0.005          | Used for soft update of the       | aka. Interpolation
           | ``target_theta``                               | target network.                   | factor in polyak aver
           |                                                |                                   | aging for target
           |                                                |                                   | networks.
        11 | ``collect.-``       float       0.1            | Used for add noise during co-     | Sample noise from dis
           | ``noise_sigma``                                | llection, through controlling     | tribution, Gaussian
           |                                                | the sigma of distribution         | process.
        12 | ``model.v_min``      float      -10            | Value of the smallest atom        |
           |                                                | in the support set.               |
        13 | ``model.v_max``      float      10             | Value of the largest atom         |
           |                                                | in the support set.               |
        14 | ``model.n_atom``     int        51             | Number of atoms in the support    |
           |                                                | set of the value distribution.    |
        15 | ``nstep``            int        3, [1, 5]      | N-step reward discount sum for    |
           |                                                | target q_value estimation         |
        16 | ``priority``         bool       True           | Whether use priority(PER)         | priority sample,
                                                                                                | update priority
        == ====================  ========    =============  =================================   =======================
    """

    config = dict(
        # (str) RL policy register name (refer to function "POLICY_REGISTRY").
        type='d4pg',
        # (bool) Whether to use cuda for network.
        cuda=False,
        # (bool type) on_policy: Determine whether on-policy or off-policy.
        # on-policy setting influences the behaviour of buffer.
        # Default False in D4PG.
        on_policy=False,
        # (bool) Whether use priority(priority sample, IS weight, update priority)
        # Default True in D4PG.
        priority=True,
        # (bool) Whether use Importance Sampling Weight to correct biased update. If True, priority must be True.
        priority_IS_weight=True,
        # (int) Number of training samples(randomly collected) in replay buffer when training starts.
        # Default 25000 in D4PG.
        random_collect_size=25000,
        # (int) N-step reward for target q_value estimation
        nstep=3,
        # (str) Action space type
        action_space='continuous',  # ['continuous', 'hybrid']
        # (bool) Whether use batch normalization for reward
        reward_batch_norm=False,
        # (bool) Whether to need policy data in process transition
        transition_with_policy_data=False,
        model=dict(
            # (float) Value of the smallest atom in the support set.
            # Default to -10.0.
            v_min=-10,
            # (float) Value of the smallest atom in the support set.
            # Default to 10.0.
            v_max=10,
            # (int) Number of atoms in the support set of the
            # value distribution. Default to 51.
            n_atom=51
        ),
        learn=dict(
            # (bool) Whether to use multi gpu
            multi_gpu=False,
            # How many updates(iterations) to train after collector's one collection.
            # Bigger "update_per_collect" means bigger off-policy.
            # collect data -> update policy-> collect data -> ...
            update_per_collect=1,
            # (int) Minibatch size for gradient descent.
            batch_size=256,
            # Learning rates for actor network(aka. policy).
            learning_rate_actor=1e-3,
            # Learning rates for critic network(aka. Q-network).
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
            actor_update_freq=1,
            # (bool) Whether to add noise on target network's action.
            # Target Policy Smoothing Regularization in original TD3 paper.
            noise=False,
        ),
        collect=dict(
            # (int) Only one of [n_sample, n_episode] should be set
            # n_sample=1,
            # (int) Cut trajectories into pieces with length "unroll_len".
            unroll_len=1,
            # It is a must to add noise during collection. So here omits "noise" and only set "noise_sigma".
            noise_sigma=0.1,
        ),
        eval=dict(evaluator=dict(eval_freq=1000, ), ),
        other=dict(
            replay_buffer=dict(
                # (int) Maximum size of replay buffer.
                replay_buffer_size=1000000,
            ),
        ),
    )

    def _init_learn(self) -> None:
        r"""
        Overview:
            Learn mode init method. Called by ``self.__init__``.
            Init actor and critic optimizers, algorithm config, main and target models.
        """
        self._priority = self._cfg.priority
        self._priority_IS_weight = self._cfg.priority_IS_weight
        # actor and critic optimizer
        self._optimizer_actor = Adam(
            self._model.actor.parameters(),
            lr=self._cfg.learn.learning_rate_actor,
        )
        self._optimizer_critic = Adam(
            self._model.critic.parameters(),
            lr=self._cfg.learn.learning_rate_critic,
        )
        self._reward_batch_norm = self._cfg.reward_batch_norm

        self._gamma = self._cfg.learn.discount_factor
        self._nstep = self._cfg.nstep
        self._actor_update_freq = self._cfg.learn.actor_update_freq

        # main and target models
        self._target_model = copy.deepcopy(self._model)
        self._target_model = model_wrap(
            self._target_model,
            wrapper_name='target',
            update_type='momentum',
            update_kwargs={'theta': self._cfg.learn.target_theta}
        )
        if self._cfg.learn.noise:
            self._target_model = model_wrap(
                self._target_model,
                wrapper_name='action_noise',
                noise_type='gauss',
                noise_kwargs={
                    'mu': 0.0,
                    'sigma': self._cfg.learn.noise_sigma
                },
                noise_range=self._cfg.learn.noise_range
            )
        self._learn_model = model_wrap(self._model, wrapper_name='base')
        self._learn_model.reset()
        self._target_model.reset()

        self._v_max = self._cfg.model.v_max
        self._v_min = self._cfg.model.v_min
        self._n_atom = self._cfg.model.n_atom

        self._forward_learn_cnt = 0  # count iterations

    def _forward_learn(self, data: dict) -> Dict[str, Any]:
        r"""
        Overview:
            Forward and backward function of learn mode.
        Arguments:
            - data (:obj:`dict`): Dict type data, including at least ['obs', 'action', 'reward', 'next_obs']
        Returns:
            - info_dict (:obj:`Dict[str, Any]`): Including at least actor and critic lr, different losses.
        """
        loss_dict = {}
        data = default_preprocess_learn(
            data,
            use_priority=self._cfg.priority,
            use_priority_IS_weight=self._cfg.priority_IS_weight,
            ignore_done=self._cfg.learn.ignore_done,
            use_nstep=True
        )
        if self._cuda:
            data = to_device(data, self._device)
        # ====================
        # critic learn forward
        # ====================
        self._learn_model.train()
        self._target_model.train()
        next_obs = data.get('next_obs')
        reward = data.get('reward')
        if self._reward_batch_norm:
            reward = (reward - reward.mean()) / (reward.std() + 1e-8)
        # current q value
        q_value = self._learn_model.forward(data, mode='compute_critic')
        q_value_dict = {}
        q_dist = q_value['distribution']
        q_value_dict['q_value'] = q_value['q_value'].mean()
        # target q value.
        with torch.no_grad():
            next_action = self._target_model.forward(next_obs, mode='compute_actor')['action']
            next_data = {'obs': next_obs, 'action': next_action}
            target_q_dist = self._target_model.forward(next_data, mode='compute_critic')['distribution']

        value_gamma = data.get('value_gamma')
        action_index = np.zeros(next_action.shape[0])
        # since the action is a scalar value, action index is set to 0 which is the only possible choice
        td_data = dist_nstep_td_data(
            q_dist, target_q_dist, action_index, action_index, reward, data['done'], data['weight']
        )
        critic_loss, td_error_per_sample = dist_nstep_td_error(
            td_data, self._gamma, self._v_min, self._v_max, self._n_atom, nstep=self._nstep, value_gamma=value_gamma
        )
        loss_dict['critic_loss'] = critic_loss
        # ================
        # critic update
        # ================
        self._optimizer_critic.zero_grad()
        for k in loss_dict:
            if 'critic' in k:
                loss_dict[k].backward()
        self._optimizer_critic.step()
        # ===============================
        # actor learn forward and update
        # ===============================
        # actor updates every ``self._actor_update_freq`` iters
        if (self._forward_learn_cnt + 1) % self._actor_update_freq == 0:
            actor_data = self._learn_model.forward(data['obs'], mode='compute_actor')
            actor_data['obs'] = data['obs']
            actor_loss = -self._learn_model.forward(actor_data, mode='compute_critic')['q_value'].mean()

            loss_dict['actor_loss'] = actor_loss
            # actor update
            self._optimizer_actor.zero_grad()
            actor_loss.backward()
            self._optimizer_actor.step()
        # =============
        # after update
        # =============
        loss_dict['total_loss'] = sum(loss_dict.values())
        self._forward_learn_cnt += 1
        self._target_model.update(self._learn_model.state_dict())
        return {
            'cur_lr_actor': self._optimizer_actor.defaults['lr'],
            'cur_lr_critic': self._optimizer_critic.defaults['lr'],
            'q_value': q_value['q_value'].mean().item(),
            'action': data['action'].mean().item(),
            'priority': td_error_per_sample.abs().tolist(),
            **loss_dict,
            **q_value_dict,
        }

    def _get_train_sample(self, traj: list) -> Union[None, List[Any]]:
        r"""
            Overview:
                Get the trajectory and the n step return data, then sample from the n_step return data
            Arguments:
                - traj (:obj:`list`): The trajectory's buffer list
            Returns:
                - samples (:obj:`dict`): The training samples generated
        """
        data = get_nstep_return_data(traj, self._nstep, gamma=self._gamma)
        return get_train_sample(data, self._unroll_len)

    def default_model(self) -> Tuple[str, List[str]]:
        return 'qac_dist', ['ding.model.template.qac_dist']

    def _monitor_vars_learn(self) -> List[str]:
        r"""
        Overview:
            Return variables' name if variables are to used in monitor.
        Returns:
            - vars (:obj:`List[str]`): Variables' name list.
        """
        ret = ['cur_lr_actor', 'cur_lr_critic', 'critic_loss', 'actor_loss', 'total_loss', 'q_value', 'action']
        return ret
