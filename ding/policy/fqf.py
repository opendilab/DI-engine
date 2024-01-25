import copy
from typing import List, Dict, Any, Tuple

import torch

from ding.model import model_wrap
from ding.rl_utils import fqf_nstep_td_data, fqf_nstep_td_error, fqf_calculate_fraction_loss
from ding.torch_utils import Adam, RMSprop, to_device
from ding.utils import POLICY_REGISTRY
from .common_utils import default_preprocess_learn
from .dqn import DQNPolicy


def compute_grad_norm(model):
    """
    Overview:
        Compute grad norm of a network's parameters.
    Arguments:
        - model (:obj:`nn.Module`): The network to compute grad norm.
    Returns:
        - grad_norm (:obj:`torch.Tensor`): The grad norm of the network's parameters.
    """
    return torch.norm(torch.stack([torch.norm(p.grad.detach(), 2.0) for p in model.parameters()]), 2.0)


@POLICY_REGISTRY.register('fqf')
class FQFPolicy(DQNPolicy):
    """
    Overview:
        Policy class of FQF (Fully Parameterized Quantile Function) algorithm, proposed in
        https://arxiv.org/pdf/1911.02140.pdf.

    Config:
        == ==================== ======== ============== ======================================== =======================
        ID Symbol               Type     Default Value  Description                              Other(Shape)
        == ==================== ======== ============== ======================================== =======================
        1  ``type``             str      fqf            | RL policy register name, refer to      | this arg is optional,
                                                        | registry ``POLICY_REGISTRY``           | a placeholder
        2  ``cuda``             bool     False          | Whether to use cuda for network        | this arg can be diff-
                                                                                                 | erent from modes
        3  ``on_policy``        bool     False          | Whether the RL algorithm is on-policy
                                                        | or off-policy
        4  ``priority``         bool     True           | Whether use priority(PER)              | priority sample,
                                                                                                 | update priority
        6  | ``other.eps``      float    0.05           | Start value for epsilon decay. It's
           | ``.start``                                 | small because rainbow use noisy net.
        7  | ``other.eps``      float    0.05           | End value for epsilon decay.
           | ``.end``
        8  | ``discount_``      float    0.97,          | Reward's future discount factor, aka.  | may be 1 when sparse
           | ``factor``                  [0.95, 0.999]  | gamma                                  | reward env
        9  ``nstep``            int      3,             | N-step reward discount sum for target
                                         [3, 5]         | q_value estimation
        10 | ``learn.update``   int      3              | How many updates(iterations) to train  | this args can be vary
           | ``per_collect``                            | after collector's one collection. Only | from envs. Bigger val
                                                        | valid in serial training               | means more off-policy
        11 ``learn.kappa``      float    /              | Threshold of Huber loss
        == ==================== ======== ============== ======================================== =======================
    """

    config = dict(
        # (str) Name of the RL policy registered in "POLICY_REGISTRY" function.
        type='fqf',
        # (bool) Flag to enable/disable CUDA for network computation.
        cuda=False,
        # (bool) Indicator of the RL algorithm's policy type (True for on-policy algorithms).
        on_policy=False,
        # (bool) Toggle for using prioritized experience replay (priority sampling and updating).
        priority=False,
        # (float) Discount factor (gamma) for calculating the future reward.
        discount_factor=0.97,
        # (int) Number of steps to consider for calculating n-step returns.
        nstep=1,
        learn=dict(
            # (int) Number of training iterations per data collection from the environment.
            update_per_collect=3,
            # (int) Size of minibatch for each update.
            batch_size=64,
            # (float) Fractional learning rate for the fraction proposal network.
            learning_rate_fraction=2.5e-9,
            # (float) Learning rate for the quantile regression network.
            learning_rate_quantile=0.00005,
            # ==============================================================
            # Algorithm-specific configurations
            # ==============================================================
            # (int) Frequency of target network updates.
            target_update_freq=100,
            # (float) Huber loss threshold (kappa in the FQF paper).
            kappa=1.0,
            # (float) Coefficient for the entropy loss term.
            ent_coef=0,
            # (bool) If set to True, the 'done' signals that indicate the end of an episode due to environment time
            # limits are disregarded. By default, this is set to False. This setting is particularly useful for tasks
            # that have a predetermined episode length, such as HalfCheetah and various other MuJoCo environments,
            # where the maximum length is capped at 1000 steps. When enabled, any 'done' signal triggered by reaching
            # the maximum episode steps will be overridden to 'False'. This ensures the accurate calculation of the
            # Temporal Difference (TD) error, using the formula `gamma * (1 - done) * next_v + reward`,
            # even when the episode surpasses the predefined step limit.
            ignore_done=False,
        ),
        collect=dict(
            # (int) Specify one of [n_sample, n_step, n_episode] for data collection.
            # n_sample=8,
            # (int) Length of trajectory segments for processing.
            unroll_len=1,
        ),
        eval=dict(),
        other=dict(
            # Epsilon-greedy strategy with a decay mechanism.
            eps=dict(
                # (str) Type of decay mechanism ['exp' for exponential, 'linear'].
                type='exp',
                # (float) Initial value of epsilon in epsilon-greedy exploration.
                start=0.95,
                # (float) Final value of epsilon after decay.
                end=0.1,
                # (int) Number of environment steps over which epsilon is decayed.
                decay=10000,
            ),
            replay_buffer=dict(
                # (int) Size of the replay buffer.
                replay_buffer_size=10000,
            ),
        ),
    )

    def default_model(self) -> Tuple[str, List[str]]:
        """
        Overview:
            Returns the default model configuration used by the FQF algorithm. ``__init__`` method will \
            automatically call this method to get the default model setting and create model.

        Returns:
            - model_info (:obj:`Tuple[str, List[str]]`): \
                Tuple containing the registered model name and model's import_names.
        """
        return 'fqf', ['ding.model.template.q_learning']

    def _init_learn(self) -> None:
        """
        Overview:
            Initialize the learn mode of policy, including related attributes and modules. For FQF, it mainly \
            contains optimizer, algorithm-specific arguments such as gamma, nstep, kappa ent_coef, main and \
            target model. This method will be called in ``__init__`` method if ``learn`` field is in ``enable_field``.

        .. note::
            For the member variables that need to be saved and loaded, please refer to the ``_state_dict_learn`` \
            and ``_load_state_dict_learn`` methods.

        .. note::
            For the member variables that need to be monitored, please refer to the ``_monitor_vars_learn`` method.

        .. note::
            If you want to set some spacial member variables in ``_init_learn`` method, you'd better name them \
            with prefix ``_learn_`` to avoid conflict with other modes, such as ``self._learn_attr1``.
        """
        self._priority = self._cfg.priority
        # Optimizer
        self._fraction_loss_optimizer = RMSprop(
            self._model.head.quantiles_proposal.parameters(),
            lr=self._cfg.learn.learning_rate_fraction,
            alpha=0.95,
            eps=0.00001
        )
        self._quantile_loss_optimizer = Adam(
            list(self._model.head.Q.parameters()) + list(self._model.head.fqf_fc.parameters()) +
            list(self._model.encoder.parameters()),
            lr=self._cfg.learn.learning_rate_quantile,
            eps=1e-2 / self._cfg.learn.batch_size
        )

        self._gamma = self._cfg.discount_factor
        self._nstep = self._cfg.nstep
        self._kappa = self._cfg.learn.kappa
        self._ent_coef = self._cfg.learn.ent_coef

        # use model_wrapper for specialized demands of different modes
        self._target_model = copy.deepcopy(self._model)
        self._target_model = model_wrap(
            self._target_model,
            wrapper_name='target',
            update_type='assign',
            update_kwargs={'freq': self._cfg.learn.target_update_freq}
        )
        self._learn_model = model_wrap(self._model, wrapper_name='argmax_sample')
        self._learn_model.reset()
        self._target_model.reset()

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
                For FQF, each element in list is a dict containing at least the following keys: \
                ['obs', 'action', 'reward', 'next_obs'].
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
        # Data preprocessing operations, such as stack data, cpu to cuda device
        data = default_preprocess_learn(
            data, use_priority=self._priority, ignore_done=self._cfg.learn.ignore_done, use_nstep=True
        )
        if self._cuda:
            data = to_device(data, self._device)
        # ====================
        # Q-learning forward
        # ====================
        self._learn_model.train()
        self._target_model.train()
        # Current q value (main model)
        ret = self._learn_model.forward(data['obs'])
        logit = ret['logit']  # [batch, action_dim(64)]
        q_value = ret['q']  # [batch, num_quantiles, action_dim(64)]
        quantiles = ret['quantiles']  # [batch, num_quantiles+1]
        quantiles_hats = ret['quantiles_hats']  # [batch, num_quantiles], requires_grad = False
        q_tau_i = ret['q_tau_i']  # [batch_size, num_quantiles-1, action_dim(64)]
        entropies = ret['entropies']  # [batch, 1]

        # Target q value
        with torch.no_grad():
            target_q_value = self._target_model.forward(data['next_obs'])['q']
            # Max q value action (main model)
            target_q_action = self._learn_model.forward(data['next_obs'])['action']

        data_n = fqf_nstep_td_data(
            q_value, target_q_value, data['action'], target_q_action, data['reward'], data['done'], quantiles_hats,
            data['weight']
        )
        value_gamma = data.get('value_gamma')
        entropy_loss = -self._ent_coef * entropies.mean()
        fraction_loss = fqf_calculate_fraction_loss(q_tau_i.detach(), q_value, quantiles, data['action']) + entropy_loss
        quantile_loss, td_error_per_sample = fqf_nstep_td_error(
            data_n, self._gamma, nstep=self._nstep, kappa=self._kappa, value_gamma=value_gamma
        )

        # ====================
        # fraction_proposal network update
        # ====================
        self._fraction_loss_optimizer.zero_grad()
        fraction_loss.backward(retain_graph=True)
        if self._cfg.multi_gpu:
            self.sync_gradients(self._learn_model)
        with torch.no_grad():
            total_norm_quantiles_proposal = compute_grad_norm(self._model.head.quantiles_proposal)
        self._fraction_loss_optimizer.step()

        # ====================
        # Q-learning update
        # ====================
        self._quantile_loss_optimizer.zero_grad()
        quantile_loss.backward()
        if self._cfg.multi_gpu:
            self.sync_gradients(self._learn_model)
        with torch.no_grad():
            total_norm_Q = compute_grad_norm(self._model.head.Q)
            total_norm_fqf_fc = compute_grad_norm(self._model.head.fqf_fc)
            total_norm_encoder = compute_grad_norm(self._model.encoder)
        self._quantile_loss_optimizer.step()

        # =============
        # after update
        # =============
        self._target_model.update(self._learn_model.state_dict())
        return {
            'cur_lr_fraction_loss': self._fraction_loss_optimizer.defaults['lr'],
            'cur_lr_quantile_loss': self._quantile_loss_optimizer.defaults['lr'],
            'logit': logit.mean().item(),
            'fraction_loss': fraction_loss.item(),
            'quantile_loss': quantile_loss.item(),
            'total_norm_quantiles_proposal': total_norm_quantiles_proposal,
            'total_norm_Q': total_norm_Q,
            'total_norm_fqf_fc': total_norm_fqf_fc,
            'total_norm_encoder': total_norm_encoder,
            'priority': td_error_per_sample.abs().tolist(),
            # Only discrete action satisfying len(data['action'])==1 can return this and draw histogram on tensorboard.
            '[histogram]action_distribution': data['action'],
            '[histogram]quantiles_hats': quantiles_hats[0],  # quantiles_hats.requires_grad = False
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
            'cur_lr_fraction_loss', 'cur_lr_quantile_loss', 'logit', 'fraction_loss', 'quantile_loss',
            'total_norm_quantiles_proposal', 'total_norm_Q', 'total_norm_fqf_fc', 'total_norm_encoder'
        ]

    def _state_dict_learn(self) -> Dict[str, Any]:
        """
        Overview:
            Return the state_dict of learn mode, usually including model and optimizer.
        Returns:
            - state_dict (:obj:`Dict[str, Any]`): The dict of current policy learn state, for saving and restoring.
        """
        return {
            'model': self._learn_model.state_dict(),
            'target_model': self._target_model.state_dict(),
            'optimizer_fraction_loss': self._fraction_loss_optimizer.state_dict(),
            'optimizer_quantile_loss': self._quantile_loss_optimizer.state_dict(),
        }

    def _load_state_dict_learn(self, state_dict: Dict[str, Any]) -> None:
        """
        Overview:
            Load the state_dict variable into policy learn mode.
        Arguments:
            - state_dict (:obj:`Dict[str, Any]`): The dict of policy learn state saved before.

        .. tip::
            If you want to only load some parts of model, you can simply set the ``strict`` argument in \
            load_state_dict to ``False``, or refer to ``ding.torch_utils.checkpoint_helper`` for more \
            complicated operation.
        """
        self._learn_model.load_state_dict(state_dict['model'])
        self._target_model.load_state_dict(state_dict['target_model'])
        self._fraction_loss_optimizer.load_state_dict(state_dict['optimizer_fraction_loss'])
        self._quantile_loss_optimizer.load_state_dict(state_dict['optimizer_quantile_loss'])
