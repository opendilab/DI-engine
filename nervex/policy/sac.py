from typing import List, Dict, Any, Tuple, Union, Optional
from collections import namedtuple, deque
import sys
import os
import numpy as np
import torch
import copy
import torch.nn.functional as F

from nervex.torch_utils import Adam, to_device
from nervex.data import default_collate, default_decollate
from nervex.rl_utils import v_1step_td_data, v_1step_td_error, Adder
from nervex.model import model_wrap
from nervex.utils import POLICY_REGISTRY
from .base_policy import Policy
from .common_utils import default_preprocess_learn


@POLICY_REGISTRY.register('sac')
class SACPolicy(Policy):
    r"""
       Overview:
           Policy class of SAC algorithm.
       """

    config = dict(
        # (str) RL policy register name (refer to function "POLICY_REGISTRY").
        tyep='sac',
        # (bool) Whether to use cuda for network.
        cuda=False,
        # (str type) policy_type: Determine the version of sac to use.
        # policy_type in ['sac_v1', 'sac_v2']
        # sac_v1: learns a value function, soft q function, and actor like the original SAC paper (arXiv 1801.01290).
        # using sac_v1 needs to set learning_rate_value, learning_rate_q and learning_rate_policy in `cfg.policy.learn`.
        # sac_v2: learns soft q function and actor.
        # Note that: Please consistent with the model type setting.
        # policy_type='sac_v2',
        # import_names=['nervex.policy.sac'],
        # (bool type) on_policy: Determine whether on-policy or off-policy.
        # on-policy setting influences the behaviour of buffer.
        # Please use False in sac.
        on_policy=False,
        # (bool type) priority: Determine whether to use priority in buffer sample.
        # Please use False in sac.
        priority=False,
        # (bool) Whether use Importance Sampling Weight to correct biased update. If True, priority must be True.
        priority_IS_weight=False,
        # (int) Number of training samples(randomly collected) in replay buffer when training starts.
        random_collect_size=2000,
        model=dict(
            obs_shape=17,
            action_shape=6,
            policy_embedding_size=256,
            # (int type) value_embedding_size: linear layer size for value network.
            # `value_embedding_size` should be initialized, when you use `sac_v1` and model.
            # Default to 256 when value_network is ture.
            value_embedding_size=256,
            soft_q_embedding_size=256,

            # (bool type) twin_q: Determine whether to use double-soft-q-net for target q computation.
            # Please refer to TD3 about Clipped Double-Q Learning trick, which learns two Q-functions instead of one .
            # Default to True.
            twin_q=True,

            # (bool type) value_network: Determine whether to use value network as the
            # original SAC paper (arXiv 1801.01290).
            # using value_network needs to set learning_rate_value, learning_rate_q,
            # and learning_rate_policy in `cfg.policy.learn`.
            # Default to False.
            value_network=False,
        ),
        learn=dict(
            multi_gpu=False,
            update_per_collect=1,
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
            # Default to 3e-4 in sac_v1.
            learning_rate_value=3e-4,

            # (float type) learning_rate_alpha: Learning rate for auto temperature parameter `\alpha`.
            # Default to 3e-4.
            learning_rate_alpha=3e-4,
            # (float type) target_theta: Used for soft update of the target network.
            # Default to 0.005.
            target_theta=0.005,
            discount_factor=0.99,

            # (float type) alpha: Entropy regularization coefficient.
            # Please check out the original SAC paper (arXiv 1801.01290): Eq 1 for more details.
            # If is_auto_alpha is set  to `True`, alpha is initialization for auto `\alpha`.
            # Default to 0.2.
            alpha=0.2,

            # (bool type) reparameterization: Determine whether to use reparameterization trick.
            # Please check out the original SAC paper (arXiv 1801.01290): Eq 11 for more details.
            # Default to True.
            reparameterization=True,

            # (bool type) is_auto_alpha: Determine whether to use auto temperature parameter `\alpha` .
            # Temperature parameter determines the relative importance of the entropy term against the reward.
            # Please check out the original SAC paper (arXiv 1801.01290): Eq 1 for more details.
            # Default to False.
            # Note that: Using auto alpha needs to set learning_rate_alpha in `cfg.policy.learn`.
            is_auto_alpha=True,
            # (bool) Whether ignore done(usually for max step termination env. e.g. pendulum)
            ignore_done=False,
        ),
        collect=dict(
            # You can use either "n_sample" or "n_episode" in actor.collect.
            # Get "n_sample" samples per collect.
            # Default n_sample to 1.
            # n_sample=1,
            # (int) Cut trajectories into pieces with length "unroll_len".
            unroll_len=1,
            # (float) The std of noise for exploration
            noise_sigma=0.2,
        ),
        eval=dict(),
        other=dict(
            replay_buffer=dict(
                # (int type) replay_buffer_size: Max size of replay buffer.
                replay_buffer_size=1000000,
                # (int type) replay_start_size: Number of experiences in replay buffer
                # when training begins. Default to 10000.
                replay_buffer_start_size=10000,
                # (int type) max_use: Max use times of one data in the buffer.
                # Data will be removed once used for too many times.
                # Default to infinite.
                # max_use=256,
            ),
        ),
    )
    r"""
    Overview:
        Policy class of SAC algorithm.
    """

    def _init_learn(self) -> None:
        r"""
        Overview:
            Learn mode init method. Called by ``self.__init__``.
            Init q, value and policy's optimizers, algorithm config, main and target models.
        """
        # Init
        self._priority = self._cfg.priority
        self._priority_IS_weight = self._cfg.priority_IS_weight
        self._value_network = self._model._value_network
        self._twin_q = self._model._twin_q

        # Optimizers
        if self._value_network:
            self._optimizer_value = Adam(
                self._model.value_net.parameters(),
                lr=self._cfg.learn.learning_rate_value,
            )
        self._optimizer_q = Adam(
            self._model.q_net.parameters(),
            lr=self._cfg.learn.learning_rate_q,
        )
        self._optimizer_policy = Adam(
            self._model.policy_net.parameters(),
            lr=self._cfg.learn.learning_rate_policy,
        )

        # Algorithm config
        self._gamma = self._cfg.learn.discount_factor
        # Init auto alpha
        if self._cfg.learn.is_auto_alpha:
            self._target_entropy = -np.prod(self._cfg.model.action_shape)
            self._log_alpha = torch.log(torch.tensor([self._cfg.learn.alpha]))
            self._log_alpha = self._log_alpha.to(device='cuda' if self._cuda else 'cpu').requires_grad_()
            self._alpha_optim = torch.optim.Adam([self._log_alpha], lr=self._cfg.learn.learning_rate_alpha)
            self._is_auto_alpha = True
            assert self._log_alpha.shape == torch.Size([1]) and self._log_alpha.requires_grad
            self._alpha = self._log_alpha.detach().exp()
        else:
            self._alpha = torch.tensor(self._cfg.learn.alpha, requires_grad=False)
            self._is_auto_alpha = False
        self._reparameterization = self._cfg.learn.reparameterization

        # Main and target models
        self._target_model = copy.deepcopy(self._model)
        self._target_model = model_wrap(
            self._target_model,
            wrapper_name='target',
            update_type='momentum',
            update_kwargs={'theta': self._cfg.learn.target_theta}
        )
        self._learn_model = model_wrap(self._model, wrapper_name='base')
        self._learn_model.reset()
        self._target_model.reset()

        self._forward_learn_cnt = 0

    def _forward_learn(self, data: dict) -> Dict[str, Any]:
        r"""
        Overview:
            Forward and backward function of learn mode.
        Arguments:
            - data (:obj:`dict`): Dict type data, including at least ['obs', 'action', 'reward', 'next_obs']
        Returns:
            - info_dict (:obj:`Dict[str, Any]`): Including current lr and loss.
        """
        loss_dict = {}
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
        obs = data.get('obs')
        next_obs = data.get('next_obs')
        reward = data.get('reward')
        action = data.get('action')
        done = data.get('done')

        # predict q value
        q_value = self._learn_model.forward(data, mode='compute_critic', qv='q')['q_value']

        # predict target value depend self._value_network.
        if self._value_network:
            # predict v value
            v_value = self._learn_model.forward(obs, mode='compute_critic', qv='v')['v_value']
            with torch.no_grad():
                next_v_value = self._target_model.forward(next_obs, mode='compute_critic', qv='v')['v_value']
        else:
            # target q value. SARSA: first predict next action, then calculate next q value
            with torch.no_grad():
                next_data = {'obs': next_obs}
                next_action = self._learn_model.forward(data['obs'], mode='compute_actor', deterministic_eval=False)
                next_data['action'] = next_action['action']
                next_data['log_prob'] = next_action['log_prob']
                target_q_value = self._target_model.forward(next_data, mode='compute_critic', qv='q')['q_value']
                # the value of a policy according to the maximum entropy objective
                if self._twin_q:
                    # find min one as target q value
                    target_q_value = torch.min(target_q_value[0],
                                               target_q_value[1]) - self._alpha * next_data['log_prob'].squeeze(-1)
                else:
                    target_q_value = target_q_value - self._alpha * next_data['log_prob'].squeeze(-1)
        target_value = next_v_value if self._value_network else target_q_value

        # =================
        # q network
        # =================
        # compute q loss
        if self._twin_q:
            q_data0 = v_1step_td_data(q_value[0], target_value, reward, done, data['weight'])
            loss_dict['q_loss'], td_error_per_sample0 = v_1step_td_error(q_data0, self._gamma)
            q_data1 = v_1step_td_data(q_value[1], target_value, reward, done, data['weight'])
            loss_dict['q_twin_loss'], td_error_per_sample1 = v_1step_td_error(q_data1, self._gamma)
            td_error_per_sample = (td_error_per_sample0 + td_error_per_sample1) / 2
        else:
            q_data = v_1step_td_data(q_value, target_value, reward, done, data['weight'])
            loss_dict['q_loss'], td_error_per_sample = v_1step_td_error(q_data, self._gamma)

        # update q network
        self._optimizer_q.zero_grad()
        loss_dict['q_loss'].backward()
        if self._twin_q:
            loss_dict['q_twin_loss'].backward()
        self._optimizer_q.step()

        # evaluate to get action distribution
        eval_data = self._learn_model.forward(data['obs'], mode='compute_actor', deterministic_eval=False)
        mean = eval_data["mean"]
        log_std = eval_data["log_std"]
        log_prob = eval_data["log_prob"]
        eval_data['obs'] = obs
        new_q_value = self._learn_model.forward(eval_data, mode='compute_critic', qv='q')['q_value']
        if self._twin_q:
            new_q_value = torch.min(new_q_value[0], new_q_value[1])

        # =================
        # value network
        # =================
        # compute value loss
        if self._value_network:
            # new_q_value: (bs, ), log_prob: (bs, act_shape) -> target_v_value: (bs, )
            target_v_value = (new_q_value.unsqueeze(-1) - self._alpha * log_prob).mean(dim=-1)
            loss_dict['value_loss'] = F.mse_loss(v_value, target_v_value.detach())

            # update value network
            self._optimizer_value.zero_grad()
            loss_dict['value_loss'].backward()
            self._optimizer_value.step()

        # =================
        # policy network
        # =================
        # compute policy loss
        if not self._reparameterization:
            target_log_policy = new_q_value - v_value
            policy_loss = (log_prob * (log_prob - target_log_policy.unsqueeze(-1))).mean()
        else:
            policy_loss = (self._alpha * log_prob - new_q_value.unsqueeze(-1)).mean()

        loss_dict['policy_loss'] = policy_loss

        # update policy network
        self._optimizer_policy.zero_grad()
        loss_dict['policy_loss'].backward()
        self._optimizer_policy.step()

        # compute alpha loss
        if self._is_auto_alpha:
            log_prob = log_prob.detach() + self._target_entropy
            loss_dict['alpha_loss'] = -(self._log_alpha * log_prob).mean()

            self._alpha_optim.zero_grad()
            loss_dict['alpha_loss'].backward()
            self._alpha_optim.step()
            self._alpha = self._log_alpha.detach().exp()

        loss_dict['total_loss'] = sum(loss_dict.values())

        info_dict = {}
        if self._value_network:
            info_dict['cur_lr_v'] = self._optimizer_value.defaults['lr']

        # =============
        # after update
        # =============
        self._forward_learn_cnt += 1
        # target update
        self._target_model.update(self._learn_model.state_dict())
        return {
            'cur_lr_q': self._optimizer_q.defaults['lr'],
            'cur_lr_p': self._optimizer_policy.defaults['lr'],
            'priority': td_error_per_sample.abs().tolist(),
            'td_error': td_error_per_sample.detach().mean().item(),
            'alpha': self._alpha.item(),
            'target_value': target_value.detach().mean().item(),
            **info_dict,
            **loss_dict
        }

    def _state_dict_learn(self) -> Dict[str, Any]:
        ret = {
            'model': self._learn_model.state_dict(),
            'optimizer_q': self._optimizer_q.state_dict(),
            'optimizer_policy': self._optimizer_policy.state_dict(),
        }
        if self._value_network:
            ret.update({'optimizer_value': self._optimizer_value.state_dict()})
        if self._is_auto_alpha:
            ret.update({'optimizer_alpha': self._alpha_optim.state_dict()})
        return ret

    def _load_state_dict_learn(self, state_dict: Dict[str, Any]) -> None:
        self._learn_model.load_state_dict(state_dict['model'])
        self._optimizer_q.load_state_dict(state_dict['optimizer_q'])
        self._optimizer_value.load_state_dict(state_dict['optimizer_value'])
        self._optimizer_policy.load_state_dict(state_dict['optimizer_policy'])
        if self._is_auto_alpha:
            self._alpha_optim.load_state_dict(state_dict['optimizer_alpha'])

    def _init_collect(self) -> None:
        r"""
        Overview:
            Collect mode init method. Called by ``self.__init__``.
            Init traj and unroll length, adder, collect model.
            Use action noise for exploration.
        """
        self._unroll_len = self._cfg.collect.unroll_len
        self._adder = Adder(self._cuda, self._unroll_len)
        #TODO remove noise
        self._collect_model = model_wrap(
            self._model,
            wrapper_name='action_noise',
            noise_type='gauss',
            noise_kwargs={
                'mu': 0.0,
                'sigma': self._cfg.collect.noise_sigma
            },
            noise_range=None
        )
        self._collect_model.reset()

    def _forward_collect(self, data: dict) -> dict:
        r"""
        Overview:
            Forward function of collect mode.
        Arguments:
            - data (:obj:`dict`): Dict type data, including at least ['obs'].
        Returns:
            - output (:obj:`dict`): Dict type data, including at least inferred action according to input obs.
        """
        data_id = list(data.keys())
        data = default_collate(list(data.values()))
        if self._cuda:
            data = to_device(data, self._device)
        self._collect_model.eval()
        with torch.no_grad():
            output = self._collect_model.forward(data, mode='compute_actor', deterministic_eval=False)
        if self._cuda:
            output = to_device(output, 'cpu')
        output = default_decollate(output)
        return {i: d for i, d in zip(data_id, output)}

    def _process_transition(self, obs: Any, model_output: dict, timestep: namedtuple) -> dict:
        r"""
        Overview:
            Generate dict type transition data from inputs.
        Arguments:
            - obs (:obj:`Any`): Env observation
            - model_output (:obj:`dict`): Output of collect model, including at least ['action']
            - timestep (:obj:`namedtuple`): Output after env step, including at least ['obs', 'reward', 'done'] \
                (here 'obs' indicates obs after env step, i.e. next_obs).
        Return:
            - transition (:obj:`Dict[str, Any]`): Dict type transition data.
        """
        transition = {
            'obs': obs,
            'next_obs': timestep.obs,
            'action': model_output['action'],
            'reward': timestep.reward,
            'done': timestep.done,
        }
        return transition

    def _get_train_sample(self, data: deque) -> Union[None, List[Any]]:
        return self._adder.get_train_sample(data)

    def _init_eval(self) -> None:
        r"""
        Overview:
            Evaluate mode init method. Called by ``self.__init__``.
            Init eval model. Unlike learn and collect model, eval model does not need noise.
        """
        self._eval_model = model_wrap(self._model, wrapper_name='base')
        self._eval_model.reset()

    def _forward_eval(self, data: dict) -> dict:
        r"""
        Overview:
            Forward function for eval mode, similar to ``self._forward_collect``.
        Arguments:
            - data (:obj:`dict`): Dict type data, including at least ['obs'].
        Returns:
            - output (:obj:`dict`): Dict type data, including at least inferred action according to input obs.
        """
        data_id = list(data.keys())
        data = default_collate(list(data.values()))
        if self._cuda:
            data = to_device(data, self._device)
        self._eval_model.eval()
        with torch.no_grad():
            output = self._eval_model.forward(data, mode='compute_actor', deterministic_eval=True)
        if self._cuda:
            output = to_device(output, 'cpu')
        output = default_decollate(output)
        return {i: d for i, d in zip(data_id, output)}

    def default_model(self) -> Tuple[str, List[str]]:
        return 'sac', ['nervex.model.sac']

    def _monitor_vars_learn(self) -> List[str]:
        r"""
        Overview:
            Return variables' name if variables are to used in monitor.
        Returns:
            - vars (:obj:`List[str]`): Variables' name list.
        """
        q_twin = ['q_twin_loss'] if self._twin_q else []
        if self._is_auto_alpha:
            return super()._monitor_vars_learn() + [
                'alpha_loss', 'policy_loss', 'q_loss', 'cur_lr_q', 'cur_lr_p', 'target_q_value', 'td_error',
                'q_value_1', 'q_value_2', 'alpha', 'target_value'
            ] + q_twin
        else:
            return super()._monitor_vars_learn() + [
                'policy_loss', 'q_loss', 'cur_lr_q', 'cur_lr_p', 'target_q_value', 'q_value_1', 'q_value_2', 'alpha',
                'td_error', 'target_value'
            ] + q_twin
