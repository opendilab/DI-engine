from collections import namedtuple
from typing import List, Dict, Any, Tuple
import copy
from pyglet.window.key import O

import torch
import numpy as np
from ding import model
from ding.model import model_wrap
from ding.rl_utils import get_train_sample, compute_q_retraces, compute_q_opc, acer_policy_error, \
    acer_value_error, acer_trust_region_update, acer_policy_error_continuous, \
    acer_value_error_continuous, compute_q_retraces_continuous
from ding.torch_utils import Adam, to_device
from ding.utils import POLICY_REGISTRY, RunningMeanStd
from ding.utils.data import default_collate, default_decollate
from ding.policy.base_policy import Policy
from torch.distributions import Normal, Independent

EPS = 1e-8


@POLICY_REGISTRY.register('acer')
class ACERPolicy(Policy):
    r"""
    Overview:
        Policy class of ACER algorithm.

    Config:
        == ======================= ======== ============== ===================================== =======================
        ID Symbol                  Type     Default Value  Description                           Other(Shape)
        == ======================= ======== ============== ===================================== =======================
        1  ``type``                str      acer           | RL policy register name, refer to   | this arg is optional,
                                                           | registry ``POLICY_REGISTRY``        | a placeholder
        2  ``cuda``                bool     False          | Whether to use cuda for network     | this arg can be diff-
                                                           |                                     | erent from modes
        3  ``on_policy``           bool     False          | Whether the RL algorithm is
                                                           | on-policy or off-policy
        4  ``trust_region``        bool     True           | Whether the RL algorithm use trust  |
                                                           | region constraint                   |
        5  ``trust_region_value``  float    1.0            | maximum range of the trust region   |
        6  ``unroll_len``          int      32             | trajectory length to calculate
                                                           | Q retrace target
        7   ``learn.update``       int      4              | How many updates(iterations) to     | this args can be vary
            ``per_collect``                                | train after collector's one         | from envs. Bigger val
                                                           |  collection. Only                   |
                                                           | valid in serial training            | means more off-policy
        8   ``c_clip_ratio``       float    1.0            | clip ratio of importance weights    |
        == ======================= ======== ============== ===================================== =======================
    """
    unroll_len = 32
    config = dict(
        type='acer',
        cuda=False,
        # (bool) whether use on-policy training pipeline(behaviour policy and training policy are the same)
        # here we follow ppo serial pipeline, the original is False
        on_policy=False,
        priority=False,
        # (bool) Whether use Importance Sampling Weight to correct biased update. If True, priority must be True.
        priority_IS_weight=False,
        learn=dict(
            # (str) the type of gradient clip method
            grad_clip_type=None,
            # (float) max value when ACER use gradient clip
            clip_value=None,
            # (bool) Whether to use multi gpu
            multi_gpu=False,
            # (int) collect n_sample data, train model update_per_collect times
            # here we follow ppo serial pipeline
            update_per_collect=4,
            # (int) the number of data for a train iteration
            batch_size=16,
            # (float) loss weight of the value network, the weight of policy network is set to 1
            value_weight=0.5,
            # (float) loss weight of the entropy regularization, the weight of policy network is set to 1
            entropy_weight=0.0001,
            # (float) discount factor for future reward, defaults int [0, 1]
            discount_factor=0.9,
            # (float) additional discounting parameter
            lambda_=0.95,
            # (int) the trajectory length to calculate v-trace target
            unroll_len=unroll_len,
            # (float) clip ratio of importance weights
            c_clip_ratio=10,
            trust_region=True,
            trust_region_value=1.0,
            learning_rate_actor=0.0005,
            learning_rate_critic=0.0005,
            target_theta=0.01
        ),
        collect=dict(
            # (int) collect n_sample data, train model n_iteration times
            n_sample=16,
            # (int) the trajectory length to calculate v-trace target
            unroll_len=unroll_len,
            # (float) discount factor for future reward, defaults int [0, 1]
            discount_factor=0.9,
            gae_lambda=0.95,
            collector=dict(
                type='sample',
                collect_print_freq=1000,
            ),
        ),
        eval=dict(evaluator=dict(eval_freq=200, ), ),
        other=dict(replay_buffer=dict(
            replay_buffer_size=1000,
            max_use=16,
        ), ),
    )

    def _init_learn(self) -> None:
        r"""
        Overview:
            Learn mode init method. Called by ``self.__init__``.
            Initialize the optimizer, algorithm config and main model.
        """
        # Optimizer
        self._optimizer_actor = Adam(
            self._model.actor.parameters(),
            lr=self._cfg.learn.learning_rate_actor,
            grad_clip_type=self._cfg.learn.grad_clip_type,
            clip_value=self._cfg.learn.clip_value,
        )
        self._optimizer_critic = Adam(
            self._model.critic.parameters(),
            lr=self._cfg.learn.learning_rate_critic,
            grad_clip_type=self._cfg.learn.grad_clip_type,
            clip_value=self._cfg.learn.clip_value,
        )
        # TODO(pu)
        # self._optimizer_critic_v = Adam(
        #     list(self._model.critic[1].V.parameters()) + list(self._model.critic[0].parameters()),
        #     lr=self._cfg.learn.learning_rate_critic,
        # )
        # target model
        self._target_model = copy.deepcopy(self._model)
        self._target_model = model_wrap(
            self._target_model,
            wrapper_name='target',
            update_type='momentum',
            update_kwargs={'theta': self._cfg.learn.target_theta}
        )
        # learn model
        self._learn_model = model_wrap(self._model, wrapper_name='base')

        if self._cfg.continuous:
            # Learn from the practice of ppo
            # init log sigma
            if hasattr(self._model.actor[1], 'log_sigma_param'):  # self._model.actor[1]:actor_head
                torch.nn.init.constant_(self._model.actor[1].log_sigma_param, -0.5)
            for m in list(self._model.critic.modules()) + list(self._model.actor.modules()):
                if isinstance(m, torch.nn.Linear):
                    # orthogonal initialization
                    torch.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                    torch.nn.init.zeros_(m.bias)

            # do last policy layer scaling, this will make initial actions have (close to)
            # 0 mean and std, and will help boost performances,
            # see https://arxiv.org/abs/2006.05990, Fig.24 for details
            for m in self._model.actor.modules():
                if isinstance(m, torch.nn.Linear):
                    torch.nn.init.zeros_(m.bias)
                    m.weight.data.copy_(0.01 * m.weight.data)

        # if self._cfg.continuous:
        #     # Learn from the practice of sac
        #     # Weight Init for the last output layer
        #     init_w = self._cfg.learn.init_w
        #     self._model.actor[1].mu.weight.data.uniform_(-init_w, init_w)
        #     self._model.actor[1].mu.bias.data.uniform_(-init_w, init_w)
        #     if hasattr(self._model.actor[1], 'log_sigma_param'):  # self._model.actor[1]:actor_head
        #         self._model.actor[1].log_sigma_layer.weight.data.uniform_(-init_w, init_w)
        #         self._model.actor[1].log_sigma_layer.bias.data.uniform_(-init_w, init_w)

        self._action_shape = self._cfg.model.action_shape
        self._unroll_len = self._cfg.learn.unroll_len

        # Algorithm config
        self._priority = self._cfg.priority
        self._priority_IS_weight = self._cfg.priority_IS_weight
        self._value_weight = self._cfg.learn.value_weight
        self._entropy_weight = self._cfg.learn.entropy_weight
        self._gamma = self._cfg.learn.discount_factor
        # self._rho_clip_ratio = self._cfg.learn.rho_clip_ratio
        self._c_clip_ratio = self._cfg.learn.c_clip_ratio
        # self._rho_pg_clip_ratio = self._cfg.learn.rho_pg_clip_ratio
        self._use_trust_region = self._cfg.learn.trust_region
        self._trust_region_value = self._cfg.learn.trust_region_value
        # Main model
        self._learn_model.reset()
        self._target_model.reset()
        self.train_cnt = 0
        self._reward_running_norm = self._cfg.learn.reward_running_norm
        self._reward_batch_norm = self._cfg.learn.reward_batch_norm

        if self._reward_running_norm:
            self._running_mean_std = RunningMeanStd(epsilon=1e-4, device=self._device)

        self._ignore_done = self._cfg.learn.ignore_done

    def _data_preprocess_learn(self, data: List[Dict[str, Any]]):
        """
        Overview:
            Data preprocess function of learn mode.
            Convert list trajectory data to to trajectory data, which is a dict of tensors.
        Arguments:
            - data (:obj:`List[Dict[str, Any]]`): List type data, a list of data for training. Each list element is a \
            dict, whose values are torch.Tensor or np.ndarray or dict/list combinations, keys include at least 'obs',\
             'next_obs', 'logit', 'action', 'reward', 'done'
        Returns:
            - data (:obj:`dict`): Dict type data. Values are torch.Tensor or np.ndarray or dict/list combinations. \
        ReturnsKeys:
            - necessary: 'logit', 'action', 'reward', 'done', 'weight', 'obs_plus_1'.
            - optional and not used in later computation: 'obs', 'next_obs'.'IS', 'collect_iter', 'replay_unique_id', \
                'replay_buffer_idx', 'priority', 'staleness', 'use'.
        ReturnsShapes:
            - obs_plus_1 (:obj:`torch.FloatTensor`): :math:`(T * B, obs_shape)`, where T is timestep, B is batch size \
                and obs_shape is the shape of single env observation
            - logit (:obj:`torch.FloatTensor`): :math:`(T, B, N)`, where N is action dim
            - action (:obj:`torch.LongTensor`): :math:`(T, B)`
            - reward (:obj:`torch.FloatTensor`): :math:`(T+1, B)`
            - done (:obj:`torch.FloatTensor`): :math:`(T, B)`
            - weight (:obj:`torch.FloatTensor`): :math:`(T, B)`
        """
        data = default_collate(data)
        if self._cuda:
            data = to_device(data, self._device)
        data['weight'] = data.get('weight', None)
        # shape (T+1)*B,env_obs_shape
        data['obs_plus_1'] = torch.cat((data['obs'] + data['next_obs'][-1:]), dim=0)
        data['action'] = torch.cat(data['action'], dim=0).reshape(self._unroll_len, -1,
                                                                  self._action_shape)  # shape T,B, or T,B,action_shape (cont)
        data['done'] = torch.cat(data['done'], dim=0).reshape(self._unroll_len, -1).float()  # shape T,B,
        data['reward'] = torch.cat(data['reward'], dim=0).reshape(self._unroll_len, -1)  # shape T,B,
        # TODO(pu): reward norm, transform to mean 0, std 1
        import copy
        if self._reward_running_norm:
            self._running_mean_std.update(data['reward'].cpu().numpy())
            data['reward'] = (copy.deepcopy(data['reward']) - self._running_mean_std.mean) / (
                    self._running_mean_std.std + EPS)
        if self._reward_batch_norm:
            data['reward'] = (copy.deepcopy(data['reward']) - copy.deepcopy(data['reward']).mean()) / (
                    copy.deepcopy(data['reward']).std() + EPS)

        data['weight'] = torch.cat(
            data['weight'], dim=0
        ).reshape(self._unroll_len, -1) if data['weight'] else None  # shape T,B
        if self._cfg.continuous:
            # change a nested list&tensor structure to pure tensor form
            data_list = []
            for i in range(len(data['logit'])):
                list2tensor = torch.Tensor(np.array([item.cpu().numpy() for item in data['logit'][i]]))
                data_list.append(list2tensor)
            data2tensor = torch.Tensor(np.array([item.cpu().numpy() for item in data_list]))

            if self._action_shape > 1:
                # reshape the tensor from (T, 2, B, env_action_shape,) to (T, B, env_action_shape,2)
                data2tensor = data2tensor.permute(0, 2, 3, 1)
            else:
                # reshape the tensor from (T, 2, B) to (T, B, 2)
                data2tensor = data2tensor.permute(0, 2, 1)

            data['logit_mu'] = data2tensor[..., 0].reshape(self._unroll_len, -1,
                                                           self._action_shape)  # shape T,B,env_action_shape
            data['logit_sigma'] = data2tensor[..., 1].reshape(self._unroll_len, -1,
                                                              self._action_shape)  # shape T,B,env_action_shape

        else:
            data['logit'] = torch.cat(
                data['logit'], dim=0
            ).reshape(self._unroll_len, -1, self._action_shape)  # shape T,B,env_action_shape
        return data

    def _forward_learn(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        r"""
        Overview:
            Forward computation graph of learn mode(updating policy).
        Arguments:
            - data (:obj:`List[Dict[str, Any]]`): List type data, a list of data for training. Each list element is a \
            dict, whose values are torch.Tensor or np.ndarray or dict/list combinations, keys include at least 'obs',\
             'next_obs', 'logit', 'action', 'reward', 'done'
        Returns:
            - info_dict (:obj:`Dict[str, Any]`): Dict type data, a info dict indicated training result, which will be \
                recorded in text log and tensorboard, values are python scalar or a list of scalars.
        ArgumentsKeys:
            - necessary: ``obs``, ``action``, ``reward``, ``next_obs``, ``done``
            - optional: 'collect_iter', 'replay_unique_id', 'replay_buffer_idx', 'priority', 'staleness', 'use', 'IS'
        ReturnsKeys:
            - necessary: ``cur_lr_actor``, ``cur_lr_critic``, ``actor_loss`,``bc_loss``,``policy_loss``,\
                ``critic_loss``,``entropy_loss``
        """
        self.train_cnt += 1
        data = self._data_preprocess_learn(data)
        self._learn_model.train()
        current_action_data = self._learn_model.forward({'obs': data['obs_plus_1']},
                                                        mode='compute_actor')  # (T+1),B,env_action_shape
        avg_action_data = self._target_model.forward({'obs': data['obs_plus_1']}, mode='compute_actor')

        if self._cfg.continuous:
            current_mu, current_sigma, behaviour_mu, behaviour_sigma, avg_mu, avg_action_sigma, actions, rewards, weights = self._reshape_data_continuous(
                current_action_data, avg_action_data, data
            )

            # # shape (T+1),B,env_action_shape
            # current_dist = torch.distributions.normal.Normal(current_mu, current_sigma)
            # # shape T,B,env_action_shape
            # behaviour_dist = torch.distributions.normal.Normal(behaviour_mu, behaviour_sigma)
            # # shape (T+1),B,env_action_shape
            # avg_dist = torch.distributions.normal.Normal(avg_mu, avg_action_sigma)

            current_dist = Independent(Normal(current_mu, current_sigma), 1)
            behaviour_dist = Independent(Normal(behaviour_mu, behaviour_sigma), 1)
            avg_dist = Independent(Normal(avg_mu, avg_action_sigma), 1)

            # action_sample = dist.rsample(sample_shape=(sample_size,))  # in case for gradient back propagation
            # action_sample = action_sample.permute(1, 0, 2)

            # TODO: here we bruteforce generate a T+1 dimenstional target pi because the data['action'] is T dimention
            # cur_act_gen = current_dist.sample()[0].reshape(1, -1, self._action_shape)
            # avg_act_gen = avg_dist.sample()[0].reshape(1, -1, self._action_shape)
            cur_act_gen = current_dist.rsample()[-1].reshape(1, -1, self._action_shape)
            avg_act_gen = avg_dist.rsample()[-1].reshape(1, -1, self._action_shape)

            # cur_act_gen = current_dist.sample()
            # avg_act_gen = avg_dist.sample()

            data['action_pred'] = torch.stack(data['action_pred'], dim=0).reshape(self._unroll_len, -1,
                                                                                  self._action_shape)

            current_action_plus_1_pred = torch.cat((data['action_pred'], cur_act_gen),
                                                   dim=0)  # shape (T+1),B,env_action_shape
            avg_action_plus_1_pred = torch.cat((data['action_pred'], avg_act_gen),
                                               dim=0)  # shape (T+1),B,env_action_shape

            # 1. current_pi
            # log_current_pi = current_dist.log_prob(current_action_plus_1)  # shape (T+1),B,env_action_shape
            tmp = torch.tanh(current_action_plus_1_pred)
            y = 1 - tmp.pow(2) + EPS
            # keep dimension for loss computation (usually for action space is 1 env. e.g. pendulum)
            log_prob = current_dist.log_prob(current_action_plus_1_pred).unsqueeze(-1)
            log_current_pi = log_prob - torch.log(y).sum(-1, keepdim=True)

            current_pi = torch.exp(log_current_pi)  # shape (T+1),B,env_action_shape
            # current_pi = current_pi.unsqueeze(-1)  # shape T,B,1

            # 2. avg_pi
            # log_avg_pi = avg_dist.log_prob(avg_action_plus_1_pred)  # shape (T+1),B,env_action_shape
            tmp = torch.tanh(avg_action_plus_1_pred)
            y = 1 - tmp.pow(2) + EPS
            # keep dimension for loss computation (usually for action space is 1 env. e.g. pendulum)
            log_prob = avg_dist.log_prob(avg_action_plus_1_pred).unsqueeze(-1)
            log_avg_pi = log_prob - torch.log(y).sum(-1, keepdim=True)

            avg_pi = torch.exp(log_avg_pi)  # shape (T+1),B,env_action_shape

            # 3. behaviour_pi
            # log_behaviour_pi = behaviour_dist.log_prob(data['action'])  # shape T,B,env_action_shape
            # TODO(pu) data['action'] have been processed by tanh
            tmp = torch.tanh(data['action_pred'])
            y = 1 - tmp.pow(2) + EPS
            # keep dimension for loss computation (usually for action space is 1 env. e.g. pendulum)
            log_prob = behaviour_dist.log_prob(data['action_pred']).unsqueeze(-1)
            log_behaviour_pi = log_prob - torch.log(y).sum(-1, keepdim=True)

            behaviour_pi = torch.exp(log_behaviour_pi)  # shape T,B,env_action_shape

            # 4. current_pi_prime
            # shape (T+1),B, env_action_shape
            action_prime_pred = current_dist.rsample()
            # log_current_pi_prime = current_dist.log_prob(action_prime)
            # TODO(pu)
            tmp = torch.tanh(action_prime_pred)
            y = 1 - tmp.pow(2) + EPS
            log_prob = current_dist.log_prob(action_prime_pred).unsqueeze(-1)
            log_current_pi_prime = log_prob - torch.log(y).sum(-1, keepdim=True)

            current_pi_prime = torch.exp(log_current_pi_prime)  # shape T,B, env_action_shape

            # 5. behaviour_pi_prime
            # log_behaviour_pi_prime = behaviour_dist.log_prob(action_prime[0:-1, ...])
            # TODO(pu)
            tmp = torch.tanh(action_prime_pred[0:-1, ...])
            y = 1 - tmp.pow(2) + EPS
            log_prob = behaviour_dist.log_prob(action_prime_pred[0:-1, ...]).unsqueeze(-1)
            log_behaviour_pi_prime = log_prob - torch.log(y).sum(-1, keepdim=True)
            behaviour_pi_prime = torch.exp(log_behaviour_pi_prime)  # shape T,B,1

            # Reshape the tensor from (T, B, N) to (T*B, N) to match the SDN head computation
            obs_data = data['obs_plus_1'].view(-1, data['obs_plus_1'].shape[-1])  # (T+1)*B, 1
            current_action_pred = current_action_plus_1_pred.view(-1,
                                                                  current_action_plus_1_pred.shape[-1])  # (T+1)*B, 1
            current_action = torch.tanh(current_action_pred)

            q_value_data = self._learn_model.forward({'obs': obs_data, 'action': current_action},
                                                     mode='compute_critic')  # (T+1)*B,1

            #  Restore the shape from (T+1)*B, 1 to (T+1), B, 1
            q_values = q_value_data['q_value'].reshape(self._unroll_len + 1, -1, 1)  # shape (T+1),B,1
            #  Restore the shape from (T+1)*B, 1 to (T+1), B, 1
            v_values = q_value_data['v_value'].reshape(self._unroll_len + 1, -1, 1)  # shape (T+1),B,1
        else:  # discrete case

            current_logit, behaviour_logit, avg_logit, actions, rewards, weights = self._reshape_data(
                current_action_data, avg_action_data, data
            )
            q_value_data = self._learn_model.forward({'obs': data['obs_plus_1']},
                                                     mode='compute_critic')  # (T+1),B,env_action_shape
            q_values = q_value_data['q_value'].reshape(
                self._unroll_len + 1, -1, self._action_shape
            )  # shape (T+1),B,env_action_shape

            # shape (T+1),B,env_action_shape
            current_pi = torch.softmax(current_logit, dim=-1)
            # shape T,B,env_action_shape
            behaviour_pi = torch.softmax(behaviour_logit, dim=-1)
            # shape (T+1),B,env_action_shape
            avg_pi = torch.softmax(avg_logit, dim=-1)

        with torch.no_grad():
            # shape (T+1),B,1
            # v_pred = (q_values * current_pi).sum(-1).unsqueeze(-1)
            # shape T,B,env_action_shape
            ratio = current_pi[0:-1, ...] / (behaviour_pi + EPS)

            if self._cfg.continuous:
                ratio_dim = torch.pow(ratio, 1 / self._action_shape)  # T,B,env_action_shape
                # ratio_dim = ratio_dim.unsqueeze(-1)  # shape T,B,1

                ratio_prime = current_pi_prime[0:-1, ...] / (behaviour_pi_prime + EPS)  # T,B,env_action_shape
                # ratio_prime = current_pi_prime / (behaviour_pi_prime + EPS)  # T,B,env_action_shape

                # Calculate retrace
                # q_retraces = compute_q_retraces_continuous(q_values, v_pred, rewards, weights, ratio_dim, self._gamma)
                q_retraces = compute_q_retraces_continuous(q_values, v_values, rewards, weights, ratio_dim, self._gamma)

                # Calculate opc
                q_opc = compute_q_opc(q_values, v_values, rewards, actions, weights, self._gamma)

                # Reshape the tensor from (T, B, N) to (B', N) to match the SDN head computation
                obs_data = data['obs_plus_1'].view(-1, data['obs_plus_1'].shape[-1])  # (T+1)*B, 1
                action_prime_pred = action_prime_pred.view(-1, action_prime_pred.shape[-1])  # (T+1)*B, 1
                action_prime = torch.tanh(action_prime_pred)

                # Calculate q_value_data_prime
                q_value_data_prime = self._learn_model.forward({'obs': obs_data, 'action': action_prime},
                                                               mode='compute_critic')  # (T+1)*B,1
                # q_value_data_prime = self._learn_model.forward(obs_data, mode='compute_critic',
                #                                                action=action_prime, )  # (T+1)*B,1
                #  Restore the shape from (T+1)*B, 1 to (T+1), B, 1
                q_values_prime = q_value_data_prime['q_value'].reshape(
                    self._unroll_len + 1, -1, 1
                )  # shape (T+1),B,1
            else:

                # Calculate retrace
                q_retraces = compute_q_retraces(q_values, v_values, rewards, actions, weights, ratio, self._gamma)

        # the terminal states' weights are 0. it needs to be shift to count valid state
        # weights_ext = torch.ones_like(weights)
        # weights_ext[1:] = weights[0:-1]
        # weights = weights_ext
        q_retraces = q_retraces[0:-1]  # shape T,B,1
        q_opc = q_opc[0:-1]  # T,B,1
        q_values = q_values[0:-1]  # shape T,B,env_action_shape or T,B,1(cont)
        q_values_prime = q_values_prime[0:-1]  # shape T,B,1
        v_values = v_values[0:-1]  # shape T,B,1
        current_pi = current_pi[0:-1]  # shape T,B,env_action_shape
        current_pi_prime = current_pi_prime[0:-1]  # shape T,B,env_action_shape
        avg_pi = avg_pi[0:-1]  # shape T,B,env_action_shape
        total_valid = weights.sum()  # 1
        # ====================
        # policy update
        # ====================
        if self._cfg.continuous:
            actor_loss, bc_loss = acer_policy_error_continuous(
                q_values_prime, q_opc, v_values, current_pi, current_pi_prime, ratio, ratio_prime, self._c_clip_ratio
            )
            # dist_new = torch.distributions.normal.Normal(current_mu[:-1], current_sigma[:-1])
            dist_new = Independent(Normal(current_mu[:-1], current_sigma[:-1]), 1)

        else:
            actor_loss, bc_loss = acer_policy_error(
                q_values, q_retraces, v_values, current_pi, actions, ratio, self._c_clip_ratio
            )
            dist_new = torch.distributions.categorical.Categorical(probs=current_pi)

        actor_loss = actor_loss * weights.unsqueeze(-1)
        bc_loss = bc_loss * weights.unsqueeze(-1)
        entropy_loss = (dist_new.entropy() * weights).unsqueeze(-1)
        if self._cfg.continuous:
            actor_update_freq = 1
        else:
            actor_update_freq = 1

        if self.train_cnt % actor_update_freq == 0:
            total_actor_loss = (actor_loss + bc_loss + self._entropy_weight * entropy_loss).sum() / total_valid
            self._optimizer_actor.zero_grad()
            actor_gradients = torch.autograd.grad(-total_actor_loss, current_pi, retain_graph=True)

            if self._use_trust_region:
                actor_gradients = acer_trust_region_update(actor_gradients, current_pi, avg_pi,
                                                           self._trust_region_value)
            current_pi.backward(actor_gradients, retain_graph=True)
            self._optimizer_actor.step()
        else:
            total_actor_loss = torch.Tensor([0])

        # ====================
        # critic update
        # ====================

        if self._cfg.continuous:
            q_value_data = self._learn_model.forward({'obs': obs_data, 'action': current_action.clone().detach()},
                                                     mode='compute_critic')  # (T+1)*B,1
            # Restore the shape from (T+1)*B, 1 to (T+1), B, 1
            q_values = q_value_data['q_value'].reshape(self._unroll_len + 1, -1, 1)  # shape (T+1),B,1
            v_values = q_value_data['v_value'].reshape(self._unroll_len + 1, -1, 1)  # shape (T+1),B,1
            q_values = q_values[0:-1]  # shape T,B,env_action_shape or T,B,1(cont)
            v_values = v_values[0:-1]  # shape T,B,env_action_shape or T,B,1(cont)

            critic_loss = (acer_value_error_continuous(q_values, v_values, q_retraces.clone().detach(),
                                                       ratio.clone().detach()) * weights.unsqueeze(
                -1)).mean() / total_valid
        else:
            critic_loss = (acer_value_error(q_values, q_retraces, actions) * weights.unsqueeze(-1)).sum() / total_valid

        self._optimizer_critic.zero_grad()
        critic_loss.backward()
        self._optimizer_critic.step()

        # self._optimizer_critic.zero_grad()
        # critic_loss_q.backward()
        # self._optimizer_critic.step()
        #
        # self._optimizer_critic_v.zero_grad()
        # critic_loss_v.backward()
        # self._optimizer_critic_v.step()

        self._target_model.update(self._learn_model.state_dict())

        with torch.no_grad():
            kl_div = avg_pi * ((avg_pi + EPS).log() - (current_pi + EPS).log())
            kl_div = (kl_div.sum(-1) * weights).sum() / total_valid

        return {
            'cur_actor_lr': self._optimizer_actor.defaults['lr'],
            'cur_critic_lr': self._optimizer_critic.defaults['lr'],
            'actor_loss': (actor_loss.sum() / total_valid).item(),
            'bc_loss': (bc_loss.sum() / total_valid).item(),
            'entropy_loss': (entropy_loss.sum() / total_valid).item(),
            'total_actor_loss': total_actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'kl_div': kl_div.item(),
            'q_values': (q_values.sum() / total_valid).item(),
            'v_values': (v_values.sum() / total_valid).item(),
            # take the the fist timestep, the fist action dim
            'current_mu': ((current_mu[:-1] * weights.unsqueeze(-1))[0, :, 0].sum() / self.cfg.learn.batch_size).item(),
            'behaviour_mu': ((behaviour_mu * weights.unsqueeze(-1))[0, :, 0].sum() / self.cfg.learn.batch_size).item(),
            'avg_mu': ((avg_mu[:-1] * weights.unsqueeze(-1))[0, :, 0].sum() / self.cfg.learn.batch_size).item(),
        }

    def _reshape_data(
            self, action_data: Dict[str, Any], avg_action_data: Dict[str, Any],
            data: Dict[str, Any]
    ) -> Tuple[Any, Any, Any, Any, Any, Any]:
        r"""
        Overview:
            Obtain weights for loss calculating, where should be 0 for done positions
            Update values and rewards with the weight
        Arguments:
            - output (:obj:`Dict[int, Any]`): Dict type data, output of learn_model forward. \
             Values are torch.Tensor or np.ndarray or dict/list combinations,keys are value, logit.
            - data (:obj:`Dict[int, Any]`): Dict type data, input of policy._forward_learn \
             Values are torch.Tensor or np.ndarray or dict/list combinations. Keys includes at \
             least ['logit', 'action', 'reward', 'done',]
        Returns:
            - data (:obj:`Tuple[Any]`): Tuple of current_logit, behaviour_logit, actions, \
             values, rewards, weights
        ReturnsShapes:
            - current_logit (:obj:`torch.FloatTensor`): :math:`((T+1), B, Obs_Shape)`, where T is timestep,\
             B is batch size and Obs_Shape is the shape of single env observation.
            - behaviour_logit (:obj:`torch.FloatTensor`): :math:`(T, B, N)`, where N is action dim.
            - avg_action_logit (:obj:`torch.FloatTensor`): :math: `(T+1, B, N)`, where N is action dim.
            - actions (:obj:`torch.LongTensor`): :math:`(T, B)`
            - rewards (:obj:`torch.FloatTensor`): :math:`(T, B)`
            - weights (:obj:`torch.FloatTensor`): :math:`(T, B)`
        """
        current_logit = action_data['logit'].reshape(
            self._unroll_len + 1, -1, self._action_shape
        )  # shape (T+1),B,env_action_shape
        behaviour_logit = data['logit']  # shape T,B,env_action_shape
        avg_action_logit = avg_action_data['logit'].reshape(
            self._unroll_len + 1, -1, self._action_shape
        )  # shape (T+1),B,env_action_shape
        actions = data['action']  # shape T,B

        rewards = data['reward']  # shape T,B
        if self._ignore_done:
            weights = torch.ones_like(rewards)  # shape T,B
        else:
            weights_ = 1 - data['done']  # shape T,B
            weights = weights_

        return current_logit, behaviour_logit, avg_action_logit, actions, rewards, weights

    def _reshape_data_continuous(
            self, action_data: Dict[str, Any], avg_action_data: Dict[str, Any],
            data: Dict[str, Any]
    ) -> Tuple[Any, Any, Any, Any, Any, Any]:
        r"""
        Overview:
            Obtain weights for loss calculating, where should be 0 for done positions
            Update values and rewards with the weight
        Arguments:
            - output (:obj:`Dict[int, Any]`): Dict type data, output of learn_model forward. \
             Values are torch.Tensor or np.ndarray or dict/list combinations,keys are value, logit.
            - data (:obj:`Dict[int, Any]`): Dict type data, input of policy._forward_learn \
             Values are torch.Tensor or np.ndarray or dict/list combinations. Keys includes at \
             least ['logit', 'action', 'reward', 'done',]
        Returns:
            - data (:obj:`Tuple[Any]`): Tuple of current_logit, behaviour_logit, actions, \
             values, rewards, weights
        ReturnsShapes:
            - current_logit (:obj:`torch.FloatTensor`): :math:`((T+1), B, act_shape)`, where T is timestep,\
             B is batch size
            - behaviour_logit (:obj:`torch.FloatTensor`): :math:`(T, B, N)`, where N is action dim.
            - avg_action_logit (:obj:`torch.FloatTensor`): :math: `(T+1, B, N)`, where N is action dim.
            - actions (:obj:`torch.LongTensor`): :math:`(T, B)`
            - values (:obj:`torch.FloatTensor`): :math:`(T+1, B)`
            - rewards (:obj:`torch.FloatTensor`): :math:`(T, B)`
            - weights (:obj:`torch.FloatTensor`): :math:`(T, B)`
        """
        current_mu = action_data['logit'][0].reshape(
            self._unroll_len + 1, -1, self._action_shape
        )  # shape (T+1),B,env_action_shape
        current_sigma = action_data['logit'][1].reshape(
            self._unroll_len + 1, -1, self._action_shape
        )  # shape (T+1),B,env_action_shape

        behaviour_mu = data['logit_mu']  # shape T,B,env_action_shape
        behaviour_sigma = data['logit_sigma']  # shape T,B,env_action_shape

        avg_mu = avg_action_data['logit'][0].reshape(
            self._unroll_len + 1, -1, self._action_shape
        )  # shape (T+1),B,env_action_shape
        avg_action_sigma = avg_action_data['logit'][1].reshape(
            self._unroll_len + 1, -1, self._action_shape
        )  # shape (T+1),B,env_action_shape

        actions = data['action']  # shape T,B or (cont)T,B,action_shape

        rewards = data['reward']  # shape T,B

        if self._ignore_done:
            weights = torch.ones_like(rewards)  # shape T,B
        else:
            weights_ = 1 - data['done']  # shape T,B
            weights = weights_

        return current_mu, current_sigma, behaviour_mu, behaviour_sigma, avg_mu, avg_action_sigma, actions, rewards, weights

    def _state_dict_learn(self) -> Dict[str, Any]:
        r"""
        Overview:
            Return the state_dict of learn mode, usually including model and optimizer.
        Returns:
            - state_dict (:obj:`Dict[str, Any]`): the dict of current policy learn state, for saving and restoring.
        """
        return {
            'model': self._learn_model.state_dict(),
            'target_model': self._target_model.state_dict(),
            'actor_optimizer': self._optimizer_actor.state_dict(),
            'critic_optimizer': self._optimizer_critic.state_dict(),
        }

    def _load_state_dict_learn(self, state_dict: Dict[str, Any]) -> None:
        r"""
        Overview:
            Load the state_dict variable into policy learn mode.
        Arguments:
            - state_dict (:obj:`Dict[str, Any]`): the dict of policy learn state saved before.
        .. tip::
            If you want to only load some parts of model, you can simply set the ``strict`` argument in \
            load_state_dict to ``False``, or refer to ``ding.torch_utils.checkpoint_helper`` for more \
            complicated operation.
        """
        self._learn_model.load_state_dict(state_dict['model'])
        self._target_model.load_state_dict(state_dict['target_model'])
        self._optimizer_actor.load_state_dict(state_dict['actor_optimizer'])
        self._optimizer_critic.load_state_dict(state_dict['critic_optimizer'])

    def _init_collect(self) -> None:
        r"""
        Overview:
            Collect mode init method. Called by ``self.__init__``, initialize algorithm arguments and collect_model.
            For discrete action, use multinomial_sample to choose action.
            For continuous action, use normal_noisy_sample to choose action (first sample action from normal distribution,
                and then add some noise to the action via OU process)
        """
        self._collect_unroll_len = self._cfg.collect.unroll_len
        if self._cfg.continuous:
            self._collect_model = model_wrap(self._model, wrapper_name='normal_noisy_sample')
        else:
            self._collect_model = model_wrap(self._model, wrapper_name='multinomial_sample')
        self._collect_model.reset()

    def _forward_collect(self, data: Dict[int, Any]) -> Dict[int, Dict[str, Any]]:
        r"""
        Overview:
            Forward computation graph of collect mode(collect training data).
        Arguments:
            - data (:obj:`Dict[int, Any]`): Dict type data, stacked env data for predicting \
            action, values are torch.Tensor or np.ndarray or dict/list combinations,keys \
            are env_id indicated by integer.
        Returns:
            - output (:obj:`Dict[int, Dict[str,Any]]`): Dict of predicting policy_output(logit, action) for each env.
        ReturnsKeys
            - necessary: ``logit``, ``action``
        """
        data_id = list(data.keys())
        data = default_collate(list(data.values()))
        if self._cuda:
            data = to_device(data, self._device)
        self._collect_model.eval()
        with torch.no_grad():
            if self._cfg.continuous:
                output = self._collect_model.forward(self._cfg.model.noise_ratio, {'obs': data}, mode='compute_actor')
            else:
                output = self._collect_model.forward({'obs': data}, mode='compute_actor')
        if self._cuda:
            output = to_device(output, 'cpu')
        output = default_decollate(output)
        output = {i: d for i, d in zip(data_id, output)}
        return output

    def _get_train_sample(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        r"""
        Overview:
            For a given trajectory(transitions, a list of transition) data, process it into a list of sample that \
            can be used for training directly.
        Arguments:
            - data (:obj:`List[Dict[str, Any]`): The trajectory data(a list of transition), each element is the same \
                format as the return value of ``self._process_transition`` method.
        Returns:
            - samples (:obj:`dict`): List of training samples.
        .. note::
            We will vectorize ``process_transition`` and ``get_train_sample`` method in the following release version. \
            And the user can customize the this data processing procedure by overriding this two methods and collector \
            itself.
        """
        return get_train_sample(data, self._unroll_len)

    def _process_transition(self, obs: Any, policy_output: Dict[str, Any], timestep: namedtuple) -> Dict[str, Any]:
        r"""
        Overview:
               Generate dict type transition data from inputs.
        Arguments:
                - obs (:obj:`Any`): Env observation,can be torch.Tensor or np.ndarray or dict/list combinations.
                - model_output (:obj:`dict`): Output of collect model, including ['logit','action']
                - timestep (:obj:`namedtuple`): Output after env step, including at least ['obs', 'reward', 'done']\
                       (here 'obs' indicates obs after env step).
        Returns:
               - transition (:obj:`dict`): Dict type transition data, including at least ['obs','next_obs', 'logit',\
               'action','reward', 'done']
        """
        if self._cfg.continuous:
            transition = {
                'obs': obs,
                'next_obs': timestep.obs,
                'logit': policy_output['logit'],
                'action_pred': policy_output['action_pred'],  # before tanh
                'action': policy_output['action'],  # after tanh
                'reward': timestep.reward,
                'done': timestep.done,
            }
        else:
            transition = {
                'obs': obs,
                'next_obs': timestep.obs,
                'logit': policy_output['logit'],
                'action': policy_output['action'],
                'reward': timestep.reward,
                'done': timestep.done,
            }
        return transition

    def _init_eval(self) -> None:
        r"""
        Overview:
            Evaluate mode init method. Called by ``self.__init__``, initialize eval_model,
            For discrete action, use argmax_sample to choose action.
            For continuous action, we pass the mu to tanh funtion and got actions
        """
        if self._cfg.continuous:
            self._eval_model = model_wrap(self._model, wrapper_name='tanh_sample')
        else:
            self._eval_model = model_wrap(self._model, wrapper_name='argmax_sample')
        self._eval_model.reset()

    def _forward_eval(self, data: Dict[int, Any]) -> Dict[int, Any]:
        r"""
        Overview:
            Forward computation graph of eval mode(evaluate policy performance), at most cases, it is similar to \
            ``self._forward_collect``.
        Arguments:
            - data (:obj:`Dict[str, Any]`): Dict type data, stacked env data for predicting policy_output(action), \
                values are torch.Tensor or np.ndarray or dict/list combinations, keys are env_id indicated by integer.
        Returns:
            - output (:obj:`Dict[int, Any]`): The dict of predicting action for the interaction with env.
        ReturnsKeys
            - necessary: ``action``
            - optional: ``logit``

        """
        data_id = list(data.keys())
        data = default_collate(list(data.values()))
        if self._cuda:
            data = to_device(data, self._device)
        self._eval_model.eval()
        with torch.no_grad():
            if self._cfg.continuous:
                output = self._eval_model.forward({'obs': data}, tanh_squash=True, mode='compute_actor')
            else:
                output = self._eval_model.forward({'obs': data}, mode='compute_actor')
        if self._cuda:
            output = to_device(output, 'cpu')
        output = default_decollate(output)
        output = {i: d for i, d in zip(data_id, output)}
        return output

    def default_model(self) -> Tuple[str, List[str]]:
        return 'acer', ['ding.model.template.acer']

    def _monitor_vars_learn(self) -> List[str]:
        r"""
        Overview:
            Return this algorithm default model setting for demonstration.
        Returns:
            - model_info (:obj:`Tuple[str, List[str]]`): model name and mode import_names
        .. note::
            The user can define and use customized network model but must obey the same interface definition indicated \
            by import_names path. For IMPALA, ``ding.model.interface.IMPALA``
        """
        return ['actor_loss', 'bc_loss', 'entropy_loss', 'total_actor_loss', 'critic_loss', 'kl_div', 'q_values',
                'v_values', 'current_mu', 'behaviour_mu', 'avg_mu']
