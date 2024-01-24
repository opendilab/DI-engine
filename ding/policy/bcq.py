import copy
from collections import namedtuple
from typing import List, Dict, Any, Tuple

import torch
import torch.nn.functional as F

from ding.model import model_wrap
from ding.policy import Policy
from ding.rl_utils import v_1step_td_data, v_1step_td_error, get_train_sample, get_nstep_return_data
from ding.torch_utils import Adam, to_device
from ding.utils import POLICY_REGISTRY
from ding.utils.data import default_collate, default_decollate
from .common_utils import default_preprocess_learn


@POLICY_REGISTRY.register('bcq')
class BCQPolicy(Policy):
    """
    Overview:
        Policy class of BCQ (Batch-Constrained deep Q-learning) algorithm, proposed in \
        https://arxiv.org/abs/1812.02900.
    """

    config = dict(
        # (str) Name of the registered RL policy (refer to the "register_policy" function).
        type='bcq',
        # (bool) Indicates if CUDA should be used for network operations.
        cuda=False,
        # (bool) Determines whether priority sampling is used in the replay buffer. Default is False.
        priority=False,
        # (bool) If True, Importance Sampling Weight is used to correct updates. Requires 'priority' to be True.
        priority_IS_weight=False,
        # (int) Number of random samples in replay buffer before training begins. Default is 10000.
        random_collect_size=10000,
        # (int) The number of steps for calculating target q_value.
        nstep=1,
        model=dict(
            # (List[int]) Sizes of the hidden layers in the actor network.
            actor_head_hidden_size=[400, 300],
            # (List[int]) Sizes of the hidden layers in the critic network.
            critic_head_hidden_size=[400, 300],
            # (float) Maximum perturbation for BCQ. Controls exploration in action space.
            phi=0.05,
        ),
        learn=dict(
            # (int) Number of policy updates per data collection step. Higher values indicate more off-policy training.
            update_per_collect=1,
            # (int) Batch size for each gradient descent step.
            batch_size=100,
            # (float) Learning rate for the Q-network. Set to 1e-3 if `model.value_network` is True.
            learning_rate_q=3e-4,
            # (float) Learning rate for the policy network. Set to 1e-3 if `model.value_network` is True.
            learning_rate_policy=3e-4,
            # (float) Learning rate for the VAE network. Initialize if `model.vae_network` is True.
            learning_rate_vae=3e-4,
            # (bool) If set to True, the 'done' signals that indicate the end of an episode due to environment time
            # limits are disregarded. By default, this is set to False. This setting is particularly useful for tasks
            # that have a predetermined episode length, such as HalfCheetah and various other MuJoCo environments,
            # where the maximum length is capped at 1000 steps. When enabled, any 'done' signal triggered by reaching
            # the maximum episode steps will be overridden to 'False'. This ensures the accurate calculation of the
            # Temporal Difference (TD) error, using the formula `gamma * (1 - done) * next_v + reward`,
            # even when the episode surpasses the predefined step limit.
            ignore_done=False,
            # (float) Polyak averaging coefficient for the target network update. Typically small.
            target_theta=0.005,
            # (float) Discount factor for future rewards, often denoted as gamma.
            discount_factor=0.99,
            # (float) Lambda for TD(lambda) learning. Weighs the trade-off between bias and variance.
            lmbda=0.75,
            # (float) Range for uniform weight initialization in the output layer.
            init_w=3e-3,
        ),
        collect=dict(
            # (int) Length of trajectory segments for unrolling. Set to higher for longer dependencies.
            unroll_len=1,
        ),
        eval=dict(),
        other=dict(
            replay_buffer=dict(
                # (int) Maximum size of the replay buffer.
                replay_buffer_size=1000000,
            ),
        ),
    )

    def default_model(self) -> Tuple[str, List[str]]:
        """
        Overview:
            Returns the default model configuration used by the BCQ algorithm. ``__init__`` method will \
            automatically call this method to get the default model setting and create model.

        Returns:
            - model_info (:obj:`Tuple[str, List[str]]`): \
                Tuple containing the registered model name and model's import_names.
        """
        return 'bcq', ['ding.model.template.bcq']

    def _init_learn(self) -> None:
        """
        Overview:
            Initialize the learn mode of policy, including related attributes and modules. For BCQ, it mainly \
            contains optimizer, algorithm-specific arguments such as gamma, main and target model. \
            This method will be called in ``__init__`` method if ``learn`` field is in ``enable_field``.

        .. note::
            For the member variables that need to be saved and loaded, please refer to the ``_state_dict_learn`` \
            and ``_load_state_dict_learn`` methods.

        .. note::
            For the member variables that need to be monitored, please refer to the ``_monitor_vars_learn`` method.

        .. note::
            If you want to set some spacial member variables in ``_init_learn`` method, you'd better name them \
            with prefix ``_learn_`` to avoid conflict with other modes, such as ``self._learn_attr1``.
        """
        # Init
        self._priority = self._cfg.priority
        self._priority_IS_weight = self._cfg.priority_IS_weight
        self.lmbda = self._cfg.learn.lmbda
        self.latent_dim = self._cfg.model.action_shape * 2

        # Optimizers
        self._optimizer_q = Adam(
            self._model.critic.parameters(),
            lr=self._cfg.learn.learning_rate_q,
        )
        self._optimizer_policy = Adam(
            self._model.actor.parameters(),
            lr=self._cfg.learn.learning_rate_policy,
        )
        self._optimizer_vae = Adam(
            self._model.vae.parameters(),
            lr=self._cfg.learn.learning_rate_vae,
        )

        # Algorithm config
        self._gamma = self._cfg.learn.discount_factor

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

    def _forward_learn(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Overview:
            Policy forward function of learn mode (training policy and updating parameters). Forward means \
            that the policy inputs some training batch data from the replay buffer and then returns the output \
            result, including various training information such as policy_loss, value_loss, entropy_loss.
        Arguments:
            - data (:obj:`List[Dict[int, Any]]`): The input data used for policy forward, including a batch of \
                training samples. For each element in list, the key of the dict is the name of data items and the \
                value is the corresponding data. Usually, the value is torch.Tensor or np.ndarray or there dict/list \
                combinations. In the ``_forward_learn`` method, data often need to first be stacked in the batch \
                dimension by some utility functions such as ``default_preprocess_learn``. \
                For BCQ, each element in list is a dict containing at least the following keys: \
                ['obs', 'action', 'adv', 'value', 'weight'].
        Returns:
            - info_dict (:obj:`Dict[str, Any]`): The information dict that indicated training result, which will be \
                recorded in text log and tensorboard, values must be python scalar or a list of scalars. For the \
                detailed definition of the dict, refer to the code of ``_monitor_vars_learn`` method.

        .. note::
            The input value can be torch.Tensor or dict/list combinations and current policy supports all of them. \
            For the data type that not supported, the main reason is that the corresponding model does not support it. \
            You can implement your own model rather than use the default model. For more information, please raise an \
            issue in GitHub repo and we will continue to follow up.
        """
        loss_dict = {}
        # Data preprocessing operations, such as stack data, cpu to cuda device
        data = default_preprocess_learn(
            data,
            use_priority=self._priority,
            use_priority_IS_weight=self._cfg.priority_IS_weight,
            ignore_done=self._cfg.learn.ignore_done,
            use_nstep=False
        )
        if len(data.get('action').shape) == 1:
            data['action'] = data['action'].reshape(-1, 1)

        if self._cuda:
            data = to_device(data, self._device)

        self._learn_model.train()
        self._target_model.train()
        obs = data['obs']
        next_obs = data['next_obs']
        reward = data['reward']
        done = data['done']
        batch_size = obs.shape[0]

        # train_vae
        vae_out = self._model.forward(data, mode='compute_vae')
        recon, mean, log_std = vae_out['recons_action'], vae_out['mu'], vae_out['log_var']
        recons_loss = F.mse_loss(recon, data['action'])
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_std - mean ** 2 - log_std.exp(), dim=1), dim=0)
        loss_dict['recons_loss'] = recons_loss
        loss_dict['kld_loss'] = kld_loss
        vae_loss = recons_loss + 0.5 * kld_loss
        loss_dict['vae_loss'] = vae_loss
        self._optimizer_vae.zero_grad()
        vae_loss.backward()
        self._optimizer_vae.step()

        # train_critic
        q_value = self._learn_model.forward(data, mode='compute_critic')['q_value']

        with (torch.no_grad()):
            next_obs_rep = torch.repeat_interleave(next_obs, 10, 0)
            z = torch.randn((next_obs_rep.shape[0], self.latent_dim)).to(self._device).clamp(-0.5, 0.5)
            vae_action = self._model.vae.decode_with_obs(z, next_obs_rep)['reconstruction_action']
            next_action = self._target_model.forward({
                'obs': next_obs_rep,
                'action': vae_action
            }, mode='compute_actor')['action']

            next_data = {'obs': next_obs_rep, 'action': next_action}
            target_q_value = self._target_model.forward(next_data, mode='compute_critic')['q_value']
            # the value of a policy according to the maximum entropy objective
            # find min one as target q value
            target_q_value = self.lmbda * torch.min(target_q_value[0], target_q_value[1]) \
                + (1 - self.lmbda) * torch.max(target_q_value[0], target_q_value[1])
            target_q_value = target_q_value.reshape(batch_size, -1).max(1)[0].reshape(-1, 1)

        q_data0 = v_1step_td_data(q_value[0], target_q_value, reward, done, data['weight'])
        loss_dict['critic_loss'], td_error_per_sample0 = v_1step_td_error(q_data0, self._gamma)
        q_data1 = v_1step_td_data(q_value[1], target_q_value, reward, done, data['weight'])
        loss_dict['twin_critic_loss'], td_error_per_sample1 = v_1step_td_error(q_data1, self._gamma)
        td_error_per_sample = (td_error_per_sample0 + td_error_per_sample1) / 2

        self._optimizer_q.zero_grad()
        (loss_dict['critic_loss'] + loss_dict['twin_critic_loss']).backward()
        self._optimizer_q.step()

        # train_policy
        z = torch.randn((obs.shape[0], self.latent_dim)).to(self._device).clamp(-0.5, 0.5)
        sample_action = self._model.vae.decode_with_obs(z, obs)['reconstruction_action']
        input = {'obs': obs, 'action': sample_action}
        perturbed_action = self._model.forward(input, mode='compute_actor')['action']
        q_input = {'obs': obs, 'action': perturbed_action}
        q = self._learn_model.forward(q_input, mode='compute_critic')['q_value'][0]
        loss_dict['actor_loss'] = -q.mean()
        self._optimizer_policy.zero_grad()
        loss_dict['actor_loss'].backward()
        self._optimizer_policy.step()
        self._forward_learn_cnt += 1
        self._target_model.update(self._learn_model.state_dict())
        return {
            'td_error': td_error_per_sample.detach().mean().item(),
            'target_q_value': target_q_value.detach().mean().item(),
            **loss_dict
        }

    def _monitor_vars_learn(self) -> List[str]:
        """
        Overview:
            Return the necessary keys for logging the return dict of ``self._forward_learn``. The logger module, such \
            as text logger, tensorboard logger, will use these keys to save the corresponding data.
        Returns:
            - necessary_keys (:obj:`List[str]`): The list of the necessary keys to be logged.
        """
        return [
            'td_error', 'target_q_value', 'critic_loss', 'twin_critic_loss', 'actor_loss', 'recons_loss', 'kld_loss',
            'vae_loss'
        ]

    def _state_dict_learn(self) -> Dict[str, Any]:
        """
        Overview:
            Return the state_dict of learn mode, usually including model and optimizer.
        Returns:
            - state_dict (:obj:`Dict[str, Any]`): The dict of current policy learn state, for saving and restoring.
        """
        ret = {
            'model': self._learn_model.state_dict(),
            'target_model': self._target_model.state_dict(),
            'optimizer_q': self._optimizer_q.state_dict(),
            'optimizer_policy': self._optimizer_policy.state_dict(),
            'optimizer_vae': self._optimizer_vae.state_dict(),
        }
        return ret

    def _init_eval(self) -> None:
        """
        Overview:
            Initialize the eval mode of policy, including related attributes and modules.
            This method will be called in ``__init__`` method if ``eval`` field is in ``enable_field``.

        .. note::
            If you want to set some spacial member variables in ``_init_eval`` method, you'd better name them \
            with prefix ``_eval_`` to avoid conflict with other modes, such as ``self._eval_attr1``.
        """
        self._eval_model = model_wrap(self._model, wrapper_name='base')
        self._eval_model.reset()

    def _forward_eval(self, data: Dict[int, Any]) -> Dict[int, Any]:
        """
        Overview:
            Policy forward function of eval mode (evaluation policy performance by interacting with envs). Forward \
            means that the policy gets some necessary data (mainly observation) from the envs and then returns the \
            action to interact with the envs.
        Arguments:
            - data (:obj:`Dict[int, Any]`): The input data used for policy forward, including at least the obs. The \
                key of the dict is environment id and the value is the corresponding data of the env.
        Returns:
            - output (:obj:`Dict[int, Any]`): The output data of policy forward, including at least the action. The \
                key of the dict is the same as the input data, i.e., environment id.

        .. note::
            The input value can be ``torch.Tensor`` or dict/list combinations, current policy supports all of them. \
            For the data type that is not supported, the main reason is that the corresponding model does not \
            support it. You can implement your own model rather than use the default model. For more information, \
            please raise an issue in GitHub repo, and we will continue to follow up.
        """
        data_id = list(data.keys())
        data = default_collate(list(data.values()))
        if self._cuda:
            data = to_device(data, self._device)
        data = {'obs': data}
        self._eval_model.eval()
        with torch.no_grad():
            output = self._eval_model.forward(data, mode='compute_eval')
        if self._cuda:
            output = to_device(output, 'cpu')
        output = default_decollate(output)
        return {i: d for i, d in zip(data_id, output)}

    def _init_collect(self) -> None:
        """
        Overview:
            Initialize the collect mode of policy, including related attributes and modules. For BCQ, it contains the \
            collect_model to balance the exploration and exploitation with ``eps_greedy_sample`` \
             mechanism, and other algorithm-specific arguments such as gamma and nstep.
            This method will be called in ``__init__`` method if ``collect`` field is in ``enable_field``.

        .. note::
            If you want to set some spacial member variables in ``_init_collect`` method, you'd better name them \
            with prefix ``_collect_`` to avoid conflict with other modes, such as ``self._collect_attr1``.
        """
        self._unroll_len = self._cfg.collect.unroll_len
        self._gamma = self._cfg.discount_factor
        self._nstep = self._cfg.nstep
        self._collect_model = model_wrap(self._model, wrapper_name='eps_greedy_sample')
        self._collect_model.reset()

    def _get_train_sample(self, transitions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        pass

    def _forward_collect(self, data: dict, **kwargs) -> dict:
        pass

    def _process_transition(self, obs: Any, model_output: dict, timestep: namedtuple) -> dict:
        pass
