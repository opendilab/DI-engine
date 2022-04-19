from typing import List, Dict, Any, Tuple, Union
import copy
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal, Independent

from ding.torch_utils import Adam, to_device
from ding.rl_utils import v_1step_td_data, v_1step_td_error, get_train_sample
from ding.model import model_wrap
from ding.utils import POLICY_REGISTRY
from ding.utils.data import default_collate, default_decollate
from ding.policy.sac import SACPolicy
from ding.policy.common_utils import default_preprocess_learn


@POLICY_REGISTRY.register('mbsac')
class MBSACPolicy(SACPolicy):
    r"""
       Overview:
           Policy class of SAC algorithm.

       Config:
           == ====================  ========    =============  ================================= =======================
           ID Symbol                Type        Default Value  Description                       Other(Shape)
           == ====================  ========    =============  ================================= =======================
           1  ``type``              str         td3            | RL policy register name, refer  | this arg is optional,
                                                               | to registry ``POLICY_REGISTRY`` | a placeholder
           2  ``cuda``              bool        True           | Whether to use cuda for network |
           3  | ``random_``         int         10000          | Number of randomly collected    | Default to 10000 for
              | ``collect_size``                               | training samples in replay      | SAC, 25000 for DDPG/
              |                                                | buffer when training starts.    | TD3.
           4  | ``model.policy_``   int         256            | Linear layer size for policy    |
              | ``embedding_size``                             | network.                        |
           5  | ``model.soft_q_``   int         256            | Linear layer size for soft q    |
              | ``embedding_size``                             | network.                        |
           6  | ``model.value_``    int         256            | Linear layer size for value     | Defalut to None when
              | ``embedding_size``                             | network.                        | model.value_network
              |                                                |                                 | is False.
           7  | ``learn.learning``  float       3e-4           | Learning rate for soft q        | Defalut to 1e-3, when
              | ``_rate_q``                                    | network.                        | model.value_network
              |                                                |                                 | is True.
           8  | ``learn.learning``  float       3e-4           | Learning rate for policy        | Defalut to 1e-3, when
              | ``_rate_policy``                               | network.                        | model.value_network
              |                                                |                                 | is True.
           9  | ``learn.learning``  float       3e-4           | Learning rate for policy        | Defalut to None when
              | ``_rate_value``                                | network.                        | model.value_network
              |                                                |                                 | is False.
           10 | ``learn.alpha``     float       0.2            | Entropy regularization          | alpha is initiali-
              |                                                | coefficient.                    | zation for auto
              |                                                |                                 | `\alpha`, when
              |                                                |                                 | auto_alpha is True
           11 | ``learn.repara_``   bool        True           | Determine whether to use        |
              | ``meterization``                               | reparameterization trick.       |
           12 | ``learn.``          bool        False          | Determine whether to use        | Temperature parameter
              | ``auto_alpha``                                 | auto temperature parameter      | determines the
              |                                                | `\alpha`.                       | relative importance
              |                                                |                                 | of the entropy term
              |                                                |                                 | against the reward.
           13 | ``learn.-``         bool        False          | Determine whether to ignore     | Use ignore_done only
              | ``ignore_done``                                | done flag.                      | in halfcheetah env.
           14 | ``learn.-``         float       0.005          | Used for soft update of the     | aka. Interpolation
              | ``target_theta``                               | target network.                 | factor in polyak aver
              |                                                |                                 | aging for target
              |                                                |                                 | networks.
           == ====================  ========    =============  ================================= =======================
       """

    config = dict(
        # (str) RL policy register name (refer to function "POLICY_REGISTRY").
        type='sac_nstep',
        # (bool) Whether to use cuda for network.
        cuda=False,
        # (bool type) on_policy: Determine whether on-policy or off-policy.
        # on-policy setting influences the behaviour of buffer.
        # Default False in SAC.
        on_policy=False,
        # (bool type) priority: Determine whether to use priority in buffer sample.
        # Default False in SAC.
        priority=False,
        # (bool) Whether use Importance Sampling Weight to correct biased update. If True, priority must be True.
        priority_IS_weight=False,
        # (int) Number of training samples(randomly collected) in replay buffer when training starts.
        # Default 10000 in SAC.
        random_collect_size=10000,
        multi_agent=False,
        model=dict(
            # (bool type) twin_critic: Determine whether to use double-soft-q-net for target q computation.
            # Please refer to TD3 about Clipped Double-Q Learning trick, which learns two Q-functions instead of one .
            # Default to True.
            twin_critic=True,

            # (bool type) value_network: Determine whether to use value network as the
            # original SAC paper (arXiv 1801.01290).
            # using value_network needs to set learning_rate_value, learning_rate_q,
            # and learning_rate_policy in `cfg.policy.learn`.
            # Default to False.
            # value_network=False,

            # (str type) action_space: Use reparameterization trick for continous action
            action_space='reparameterization',
        ),
        learn=dict(
            value_expansion_horizon=0,
            value_expansion_norm=True,
            value_expansion_type='mve', # 'steve' or 'mve'
            value_expansion_grad_clip_norm=0,

            value_gradient_horizon=0,
            value_gradient_norm=True,
            value_gradient_grad_clip_norm=0,

            # (bool) Whether to use multi gpu
            multi_gpu=False,
            # How many updates(iterations) to train after collector's one collection.
            # Bigger "update_per_collect" means bigger off-policy.
            # collect data -> update policy-> collect data -> ...
            update_per_collect=1,
            # (int) Minibatch size for gradient descent.
            batch_size=256,

            # (float type) learning_rate_q: Learning rate for soft q network.
            # Default to 3e-4.
            # Please set to 1e-3, when model.value_network is True.
            learning_rate_q=3e-4,
            # (float type) learning_rate_policy: Learning rate for policy network.
            # Default to 3e-4.
            # Please set to 1e-3, when model.value_network is True.
            learning_rate_policy=3e-4,
            # (float type) learning_rate_value: Learning rate for value network.
            # `learning_rate_value` should be initialized, when model.value_network is True.
            # Please set to 3e-4, when model.value_network is True.
            learning_rate_value=3e-4,

            # (float type) learning_rate_alpha: Learning rate for auto temperature parameter `\alpha`.
            # Default to 3e-4.
            learning_rate_alpha=3e-4,
            # (float type) target_theta: Used for soft update of the target network,
            # aka. Interpolation factor in polyak averaging for target networks.
            # Default to 0.005.
            target_theta=0.005,
            # (float) discount factor for the discounted sum of rewards, aka. gamma.
            discount_factor=0.99,

            # (float type) alpha: Entropy regularization coefficient.
            # Please check out the original SAC paper (arXiv 1801.01290): Eq 1 for more details.
            # If auto_alpha is set  to `True`, alpha is initialization for auto `\alpha`.
            # Default to 0.2.
            alpha=0.2,

            # (bool type) auto_alpha: Determine whether to use auto temperature parameter `\alpha` .
            # Temperature parameter determines the relative importance of the entropy term against the reward.
            # Please check out the original SAC paper (arXiv 1801.01290): Eq 1 for more details.
            # Default to False.
            # Note that: Using auto alpha needs to set learning_rate_alpha in `cfg.policy.learn`.
            auto_alpha=True,
            # (bool type) log_space: Determine whether to use auto `\alpha` in log space.
            log_space=True,
            # (bool) Whether ignore done(usually for max step termination env. e.g. pendulum)
            # Note: Gym wraps the MuJoCo envs by default with TimeLimit environment wrappers.
            # These limit HalfCheetah, and several other MuJoCo envs, to max length of 1000.
            # However, interaction with HalfCheetah always gets done with False,
            # Since we inplace done==True with done==False to keep
            # TD-error accurate computation(``gamma * (1 - done) * next_v + reward``),
            # when the episode step is greater than max episode step.
            ignore_done=False,
            # (float) Weight uniform initialization range in the last output layer
            init_w=3e-3,
        ),
        collect=dict(
            # If you need the data collected by the collector to contain logit key which reflect the probability of
            # the action, you can change the key to be True.
            # In Guided cost Learning, we need to use logit to train the reward model, we change the key to be True.
            # Default collector_logit to False.
            collector_logit=False,
            # You can use either "n_sample" or "n_episode" in actor.collect.
            # Get "n_sample" samples per collect.
            # Default n_sample to 1.
            n_sample=1,
            # (int) Cut trajectories into pieces with length "unroll_len".
            unroll_len=1,
        ),
        eval=dict(
            evaluator=dict(
                # (int) Evaluate every "eval_freq" training iterations.
                eval_freq=5000,
            ),
        ),
        other=dict(
            replay_buffer=dict(
                # (int type) replay_buffer_size: Max size of replay buffer.
                replay_buffer_size=1000000,
                # (int type) max_use: Max use times of one data in the buffer.
                # Data will be removed once used for too many times.
                # Default to infinite.
                # max_use=256,
            ),
        ),
    )
    r"""
    Overview:
        # TODO
    """

    def _init_learn(self) -> None:
        r"""
        Overview:
            Learn mode init method. Called by ``self.__init__``.
            Init q, value and policy's optimizers, algorithm config, main and target models.
        """
        super()._init_learn()

        self._value_expansion_horizon = self._cfg.learn.value_expansion_horizon
        self._value_expansion_type = self._cfg.learn.value_expansion_type
        self._value_expansion_norm = self._cfg.learn.value_expansion_norm
        self._value_expansion_grad_clip_norm = self._cfg.learn.value_expansion_grad_clip_norm
        # TODO: implement steve style value expansion
        self._value_expansion_type = 'mve'

        self._value_gradient_horizon = self._cfg.learn.value_gradient_horizon
        self._value_gradient_norm = self._cfg.learn.value_gradient_norm
        self._value_gradient_grad_clip_norm = self._cfg.learn.value_gradient_grad_clip_norm

        self._history_vars = dict()
        self._history_loss = dict()

        # assert (self._value_expansion_horizon > 0 or self._value_gradient_horizon > 0) \
        #         and hasattr(self, '_env_model') and hasattr(self, '_model_env'), "_env_model missing"


    
    def _forward_learn(self, data: dict) -> Dict[str, Any]:
        r"""
        Overview:
            # TODO
        """
        mode = None
        if 'mode' in data:
            mode = data['mode']
            data = data['data']

        data = default_preprocess_learn(
            data,
            use_priority=self._priority,
            use_priority_IS_weight=self._cfg.priority_IS_weight,
            ignore_done=self._cfg.learn.ignore_done,
            use_nstep=False
        )
        if self._cuda:
            data = to_device(data, self._device)

        self._learn_model.train()
        self._target_model.train()

        if mode == 'policy' or not mode:
            self._update_policy(data['obs'])
        if mode == 'value' or not mode:
            self._update_value(data)

        if self._auto_alpha:
            self._update_temperature(data['obs'])


        self._history_vars['total_loss'] = sum(self._history_loss.values())
        self._history_vars.update(self._history_loss)

        # =============
        # after update
        # =============
        self._forward_learn_cnt += 1
        # target update
        if mode == 'value' or not mode:
            self._target_model.update(self._learn_model.state_dict())
        return self._history_vars

    
    def _forward_helper(self, obs):

        (mu, sigma) = self._learn_model.forward(obs, mode='compute_actor')['logit']
        dist = Independent(Normal(mu, sigma), 1)
        pred = dist.rsample()
        action = torch.tanh(pred)
        # keep dimension for loss computation (usually for action space is 1 env. e.g. pendulum)
        log_prob = dist.log_prob(pred) + 2 * torch.log(torch.cosh(pred)).sum(-1)

        return action, log_prob


    def _update_value(self, data):

        value_loss = 0

        obs      = data['obs']
        action   = data['action']
        next_obs = data['next_obs']
        reward   = data['reward']
        done     = data['done']

        # TODO: steve
        if self._value_expansion_type == 'mve':

            obs_list       = [obs, next_obs]
            action_list    = [action]
            reward_list    = [reward]
            done_list      = [done]
            done_mask_list = [torch.zeros_like(next_obs.sum(-1)).bool(), done.bool()]

            with torch.no_grad():
                # td-k trick

                for _ in range(self._value_expansion_horizon):
                    action, log_prob = self._forward_helper(next_obs)
                    reward, next_obs  = self._env_model.batch_predict(next_obs, action)
                    reward = reward - self._alpha * log_prob
                    done = self._model_env.termination_fn(next_obs)
                    done_mask = done_mask_list[-1] | done
                    obs_list.append(next_obs)
                    action_list.append(action)
                    reward_list.append(reward)
                    done_list.append(done)
                    done_mask_list.append(done_mask)

                action, log_prob = self._forward_helper(next_obs)
                eval_data = {'obs': next_obs, 'action': action}
                target_q_value = self._target_model.forward(eval_data, mode='compute_critic')['q_value']
                # the value of a policy according to the maximum entropy objective
                if self._twin_critic:
                    # find min one as target q value
                    target_q_value = torch.min(target_q_value[0],
                                               target_q_value[1]) - self._alpha * log_prob
                else:
                    target_q_value = target_q_value - self._alpha * log_prob

            for obs, action, reward, done, done_mask in reversed(
                    list(zip(obs_list[:-1], action_list, reward_list, done_list, done_mask_list[:-1]))):
                
                eval_data = {'obs': obs, 'action': action}
                q_value = self._learn_model.forward(eval_data, mode='compute_critic')['q_value']

                if self._twin_critic:
                    q_data0 = v_1step_td_data(q_value[0], target_q_value, reward, done.int(), 1-done_mask.int())
                    loss, td_error_per_sample0 = v_1step_td_error(q_data0, self._gamma)
                    value_loss += loss
                    q_data1 = v_1step_td_data(q_value[1], target_q_value, reward, done.int(), 1-done_mask.int())
                    loss, td_error_per_sample1 = v_1step_td_error(q_data1, self._gamma)
                    value_loss += loss
                    td_error_per_sample = (td_error_per_sample0 + td_error_per_sample1) / 2
                else:
                    q_data = v_1step_td_data(q_value, target_q_value, reward, done.int(), 1-done_mask.int())
                    loss, td_error_per_sample = v_1step_td_error(q_data, self._gamma)
                    value_loss += loss 

                target_q_value = reward + (1 - done.int()) * self._gamma * target_q_value

            if self._value_expansion_norm:
                value_loss = value_loss / (self._value_expansion_horizon + 1) 

            self._history_loss['value_loss'] = value_loss
            self._history_vars.update({
                'cur_lr_q': self._optimizer_q.defaults['lr'],
                # 'priority': td_error_per_sample.abs().tolist(),
                'td_error': td_error_per_sample.detach().mean().item(),
                've_rollout_termination_ratio': done_mask_list[-2].sum() / done_mask_list[-2].numel(),
            })

            self._optimizer_q.zero_grad()
            self._history_loss['value_loss'].backward()
            if self._value_expansion_grad_clip_norm:
                torch.nn.utils.clip_grad_norm_(
                        self._model.critic.parameters(), 
                        max_norm=self._value_expansion_grad_clip_norm,
                        error_if_nonfinite=False)
            self._optimizer_q.step()
                

    def _update_policy(self, obs):
        policy_loss = 0

        done_mask = torch.zeros_like(obs.sum(-1)).bool()
        for i in range(self._value_gradient_horizon):
            action, log_prob = self._forward_helper(obs)
            reward, obs  = self._env_model.batch_predict(obs, action)
            policy_loss += (self._gamma ** i) * (
                (1 - done_mask.int()) * (self._alpha * log_prob  - reward)).mean()
            done = self._model_env.termination_fn(obs)
            done_mask = done_mask | done

        # calculate the q value for the final state
        action, log_prob = self._forward_helper(obs)
        eval_data = {'obs': obs, 'action': action}
        new_q_value = self._learn_model.forward(eval_data, mode='compute_critic')['q_value']
        if self._twin_critic:
            new_q_value = torch.min(new_q_value[0], new_q_value[1])
        # TODO: if self._value_expansion_horizon > 0 and self._value_expansion_type = 'steve'
        #   new_q_value = new_q_value.mean(0)
        policy_loss += (self._gamma ** self._value_gradient_horizon) * (
            (1 - done_mask.int()) * (self._alpha * log_prob - new_q_value)).mean()

        if self._value_gradient_norm:
            policy_loss = policy_loss / (self._value_gradient_horizon + 1)

        self._history_loss['policy_loss'] = policy_loss
        self._history_vars.update({
            'vg_rollout_termination_ratio': done_mask.sum() / done_mask.numel(),
            'cur_lr_p': self._optimizer_policy.defaults['lr']
        })

        # update policy network
        self._optimizer_policy.zero_grad()
        self._history_loss['policy_loss'].backward()
        if self._value_gradient_grad_clip_norm:
            torch.nn.utils.clip_grad_norm_(
                    self._model.actor.parameters(), 
                    max_norm=self._value_gradient_grad_clip_norm,
                    error_if_nonfinite=False)
        self._optimizer_policy.step()


    def _update_temperature(self, obs):
        action, log_prob = self._forward_helper(obs)

        if self._log_space:
            log_prob = log_prob + self._target_entropy
            self._history_loss['alpha_loss'] = -(self._log_alpha * log_prob.detach()).mean()

            self._alpha_optim.zero_grad()
            self._history_loss['alpha_loss'].backward()
            self._alpha_optim.step()
            self._alpha = self._log_alpha.detach().exp()
        else:
            log_prob = log_prob + self._target_entropy
            self._history_loss['alpha_loss'] = -(self._alpha * log_prob.detach()).mean()

            self._alpha_optim.zero_grad()
            self._history_loss['alpha_loss'].backward()
            self._alpha_optim.step()
            self._alpha = max(0, self._alpha)

        self._history_vars.update({'alpha': self._alpha.item()})


    def _monitor_vars_learn(self) -> List[str]:
        r"""
        Overview:
            Return variables' name if variables are to used in monitor.
        Returns:
            - vars (:obj:`List[str]`): Variables' name list.
        """

        return [
            'alpha_loss',
            'policy_loss',
            'value_loss',
            'cur_lr_q',
            'cur_lr_p',
            'alpha',
            'td_error',
            've_rollout_termination_ratio',
            'vg_rollout_termination_ratio',
        ]

