from typing import Dict, Any, List, Tuple
from collections import namedtuple
from easydict import EasyDict

import torch
import torch.nn.functional as F

from ding.model import model_wrap
from ding.torch_utils import to_device
from ding.utils.data import default_collate, default_decollate
from ding.utils import POLICY_REGISTRY
from .bc import BehaviourCloningPolicy
from ding.model.template.ebm import create_stochastic_optimizer
from ding.model.template.ebm import StochasticOptimizer, MCMC, AutoRegressiveDFO
from ding.torch_utils import unsqueeze_repeat
from ding.utils import EasyTimer


@POLICY_REGISTRY.register('ibc')
class IBCPolicy(BehaviourCloningPolicy):
    r"""
    Overview:
        Policy class of IBC (Implicit Behavior Cloning), proposed in https://arxiv.org/abs/2109.00137.pdf.

    .. note::
        The code is adapted from the pytorch version of IBC https://github.com/kevinzakka/ibc, which only supports the \
        derivative-free optimization (dfo) variants. This implementation moves a step forward and supports all \
        variants of energy-based model mentioned in the paper (dfo, autoregressive dfo, and mcmc).
    """

    config = dict(
        # (str) The policy type. 'ibc' refers to Implicit Behavior Cloning.
        type='ibc',
        # (bool) Whether to use CUDA for training. False means CPU will be used.
        cuda=False,
        # (bool) If True, the policy will operate on-policy. Here it's False, indicating off-policy.
        on_policy=False,
        # (bool) Whether the action space is continuous. True for continuous action space.
        continuous=True,
        # (dict) Configuration for the model, including stochastic optimization settings.
        model=dict(
            # (dict) Configuration for the stochastic optimization, specifying the type of optimizer.
            stochastic_optim=dict(
                # (str) The type of stochastic optimizer. 'mcmc' refers to Markov Chain Monte Carlo methods.
                type='mcmc',
            ),
        ),
        # (dict) Configuration for the learning process.
        learn=dict(
            # (int) The number of training epochs.
            train_epoch=30,
            # (int) The size of batches used during training.
            batch_size=256,
            # (dict) Configuration for the optimizer used during training.
            optim=dict(
                # (float) The learning rate for the optimizer.
                learning_rate=1e-5,
                # (float) The weight decay regularization term for the optimizer.
                weight_decay=0.0,
                # (float) The beta1 hyperparameter for the AdamW optimizer.
                beta1=0.9,
                # (float) The beta2 hyperparameter for the AdamW optimizer.
                beta2=0.999,
            ),
        ),
        # (dict) Configuration for the evaluation process.
        eval=dict(
            # (dict) Configuration for the evaluator.
            evaluator=dict(
                # (int) The frequency of evaluations during training, in terms of number of training steps.
                eval_freq=10000,
            ),
        ),
    )

    def default_model(self) -> Tuple[str, List[str]]:
        """
        Overview:
            Returns the default model configuration used by the IBC algorithm. ``__init__`` method will \
            automatically call this method to get the default model setting and create model.

        Returns:
            - model_info (:obj:`Tuple[str, List[str]]`): \
                Tuple containing the registered model name and model's import_names.
        """
        return 'ebm', ['ding.model.template.ebm']

    def _init_learn(self) -> None:
        """
        Overview:
            Initialize the learn mode of policy, including related attributes and modules. For IBC, it mainly \
            contains optimizer and main model. \
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
        self._timer = EasyTimer(cuda=self._cfg.cuda)
        self._sync_timer = EasyTimer(cuda=self._cfg.cuda)
        optim_cfg = self._cfg.learn.optim
        self._optimizer = torch.optim.AdamW(
            self._model.parameters(),
            lr=optim_cfg.learning_rate,
            weight_decay=optim_cfg.weight_decay,
            betas=(optim_cfg.beta1, optim_cfg.beta2),
        )
        self._stochastic_optimizer: StochasticOptimizer = \
            create_stochastic_optimizer(self._device, self._cfg.model.stochastic_optim)
        self._learn_model = model_wrap(self._model, 'base')
        self._learn_model.reset()

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
                For IBC, each element in list is a dict containing at least the following keys: \
                ['obs', 'action'].
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
        with self._timer:
            data = default_collate(data)
            if self._cuda:
                data = to_device(data, self._device)
            self._learn_model.train()

            loss_dict = dict()

            # obs: (B, O)
            # action: (B, A)
            obs, action = data['obs'], data['action']
            # When action/observation space is 1, the action/observation dimension will
            # be squeezed in the first place, therefore unsqueeze there to make the data
            # compatible with the ibc pipeline.
            if len(obs.shape) == 1:
                obs = obs.unsqueeze(-1)
            if len(action.shape) == 1:
                action = action.unsqueeze(-1)

            # N refers to the number of negative samples, i.e. self._stochastic_optimizer.inference_samples.
            # (B, N, O), (B, N, A)
            obs, negatives = self._stochastic_optimizer.sample(obs, self._learn_model)

            # (B, N+1, A)
            targets = torch.cat([action.unsqueeze(dim=1), negatives], dim=1)
            # (B, N+1, O)
            obs = torch.cat([obs[:, :1], obs], dim=1)

            permutation = torch.rand(targets.shape[0], targets.shape[1]).argsort(dim=1)
            targets = targets[torch.arange(targets.shape[0]).unsqueeze(-1), permutation]

            # (B, )
            ground_truth = (permutation == 0).nonzero()[:, 1].to(self._device)

            # (B, N+1) for ebm
            # (B, N+1, A) for autoregressive ebm
            energy = self._learn_model.forward(obs, targets)

            logits = -1.0 * energy
            if isinstance(self._stochastic_optimizer, AutoRegressiveDFO):
                # autoregressive case
                # (B, A)
                ground_truth = unsqueeze_repeat(ground_truth, logits.shape[-1], -1)
            loss = F.cross_entropy(logits, ground_truth)
            loss_dict['ebm_loss'] = loss.item()

            if isinstance(self._stochastic_optimizer, MCMC):
                grad_penalty = self._stochastic_optimizer.grad_penalty(obs, targets, self._learn_model)
                loss += grad_penalty
                loss_dict['grad_penalty'] = grad_penalty.item()
            loss_dict['total_loss'] = loss.item()

            self._optimizer.zero_grad()
            loss.backward()
            with self._sync_timer:
                if self._cfg.multi_gpu:
                    self.sync_gradients(self._learn_model)
            sync_time = self._sync_timer.value
            self._optimizer.step()

        total_time = self._timer.value

        return {
            'total_time': total_time,
            'sync_time': sync_time,
            **loss_dict,
        }

    def _monitor_vars_learn(self) -> List[str]:
        """
        Overview:
            Return the necessary keys for logging the return dict of ``self._forward_learn``. The logger module, such \
            as text logger, tensorboard logger, will use these keys to save the corresponding data.
        Returns:
            - necessary_keys (:obj:`List[str]`): The list of the necessary keys to be logged.
        """
        if isinstance(self._stochastic_optimizer, MCMC):
            return ['total_loss', 'ebm_loss', 'grad_penalty', 'total_time', 'sync_time']
        else:
            return ['total_loss', 'ebm_loss', 'total_time', 'sync_time']

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
        tensor_input = isinstance(data, torch.Tensor)
        if not tensor_input:
            data_id = list(data.keys())
            data = default_collate(list(data.values()))

        if self._cuda:
            data = to_device(data, self._device)

        self._eval_model.eval()
        output = self._stochastic_optimizer.infer(data, self._eval_model)
        output = dict(action=output)

        if self._cuda:
            output = to_device(output, 'cpu')
        if tensor_input:
            return output
        else:
            output = default_decollate(output)
            return {i: d for i, d in zip(data_id, output)}

    def set_statistic(self, statistics: EasyDict) -> None:
        """
        Overview:
            Set the statistics of the environment, including the action space and the observation space.
        Arguments:
            - statistics (:obj:`EasyDict`): The statistics of the environment. For IBC, it contains at least the \
                following keys: ['action_bounds'].
        """
        self._stochastic_optimizer.set_action_bounds(statistics.action_bounds)

    # =================================================================== #
    # Implicit Behavioral Cloning does not need `collect`-related functions
    # =================================================================== #
    def _init_collect(self):
        raise NotImplementedError

    def _forward_collect(self, data: Dict[int, Any], eps: float) -> Dict[int, Any]:
        raise NotImplementedError

    def _process_transition(self, obs: Any, model_output: dict, timestep: namedtuple) -> dict:
        raise NotImplementedError

    def _get_train_sample(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        raise NotImplementedError
