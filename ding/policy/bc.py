import math
import torch
import torch.nn as nn
import copy
from torch.optim import Adam, SGD, AdamW
from torch.optim.lr_scheduler import LambdaLR
import logging
from typing import List, Dict, Any, Tuple, Union, Optional
from collections import namedtuple
from easydict import EasyDict
from ding.policy import Policy
from ding.model import model_wrap
from ding.torch_utils import to_device, to_list
from ding.utils import EasyTimer
from ding.utils.data import default_collate, default_decollate
from ding.rl_utils import get_nstep_return_data, get_train_sample
from ding.utils import POLICY_REGISTRY
from ding.torch_utils.loss.cross_entropy_loss import LabelSmoothCELoss


@POLICY_REGISTRY.register('bc')
class BehaviourCloningPolicy(Policy):
    """
    Overview:
        Behaviour Cloning (BC) policy class, which supports both discrete and continuous action space. \
        The policy is trained by supervised learning, and the data is a offline dataset collected by expert.
    """

    config = dict(
        type='bc',
        cuda=False,
        on_policy=False,
        continuous=False,
        action_shape=19,
        learn=dict(
            update_per_collect=1,
            batch_size=32,
            learning_rate=1e-5,
            lr_decay=False,
            decay_epoch=30,
            decay_rate=0.1,
            warmup_lr=1e-4,
            warmup_epoch=3,
            optimizer='SGD',
            momentum=0.9,
            weight_decay=1e-4,
            ce_label_smooth=False,
            show_accuracy=False,
            tanh_mask=False,  # if actions always converge to 1 or -1, use this.
        ),
        collect=dict(
            unroll_len=1,
            noise=False,
            noise_sigma=0.2,
            noise_range=dict(
                min=-0.5,
                max=0.5,
            ),
        ),
        eval=dict(),  # for compatibility
    )

    def default_model(self) -> Tuple[str, List[str]]:
        """
        Overview:
            Return this algorithm default neural network model setting for demonstration. ``__init__`` method will \
            automatically call this method to get the default model setting and create model.
        Returns:
            - model_info (:obj:`Tuple[str, List[str]]`): The registered model name and model's import_names.

        .. note::
            The user can define and use customized network model but must obey the same inferface definition indicated \
            by import_names path. For example about discrete BC, its registered name is ``discrete_bc`` and the \
            import_names is ``ding.model.template.bc``.
        """
        if self._cfg.continuous:
            return 'continuous_bc', ['ding.model.template.bc']
        else:
            return 'discrete_bc', ['ding.model.template.bc']

    def _init_learn(self) -> None:
        """
        Overview:
            Initialize the learn mode of policy, including related attributes and modules. For BC, it mainly contains \
            optimizer, algorithm-specific arguments such as lr_scheduler, loss, etc. \
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
        assert self._cfg.learn.optimizer in ['SGD', 'Adam'], self._cfg.learn.optimizer
        if self._cfg.learn.optimizer == 'SGD':
            self._optimizer = SGD(
                self._model.parameters(),
                lr=self._cfg.learn.learning_rate,
                weight_decay=self._cfg.learn.weight_decay,
                momentum=self._cfg.learn.momentum
            )
        elif self._cfg.learn.optimizer == 'Adam':
            if self._cfg.learn.weight_decay is None:
                self._optimizer = Adam(
                    self._model.parameters(),
                    lr=self._cfg.learn.learning_rate,
                )
            else:
                self._optimizer = AdamW(
                    self._model.parameters(),
                    lr=self._cfg.learn.learning_rate,
                    weight_decay=self._cfg.learn.weight_decay
                )
        if self._cfg.learn.lr_decay:

            def lr_scheduler_fn(epoch):
                if epoch <= self._cfg.learn.warmup_epoch:
                    return self._cfg.learn.warmup_lr / self._cfg.learn.learning_rate
                else:
                    ratio = (epoch - self._cfg.learn.warmup_epoch) // self._cfg.learn.decay_epoch
                    return math.pow(self._cfg.learn.decay_rate, ratio)

            self._lr_scheduler = LambdaLR(self._optimizer, lr_scheduler_fn)
        self._timer = EasyTimer(cuda=True)
        self._learn_model = model_wrap(self._model, 'base')
        self._learn_model.reset()

        if self._cfg.continuous:
            if self._cfg.loss_type == 'l1_loss':
                self._loss = nn.L1Loss()
            elif self._cfg.loss_type == 'mse_loss':
                self._loss = nn.MSELoss()
            else:
                raise KeyError("not support loss type: {}".format(self._cfg.loss_type))
        else:
            if not self._cfg.learn.ce_label_smooth:
                self._loss = nn.CrossEntropyLoss()
            else:
                self._loss = LabelSmoothCELoss(0.1)

    def _forward_learn(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Overview:
            Policy forward function of learn mode (training policy and updating parameters). Forward means \
            that the policy inputs some training batch data from the replay buffer and then returns the output \
            result, including various training information such as loss and time.
        Arguments:
            - data (:obj:`List[Dict[int, Any]]`): The input data used for policy forward, including a batch of \
                training samples. For each element in list, the key of the dict is the name of data items and the \
                value is the corresponding data. Usually, the value is torch.Tensor or np.ndarray or there dict/list \
                combinations. In the ``_forward_learn`` method, data often need to first be stacked in the batch \
                dimension by some utility functions such as ``default_preprocess_learn``. \
                For BC, each element in list is a dict containing at least the following keys: ``obs``, ``action``.
        Returns:
            - info_dict (:obj:`Dict[str, Any]`): The information dict that indicated training result, which will be \
                recorded in text log and tensorboard, values must be python scalar or a list of scalars. For the \
                detailed definition of the dict, refer to the code of ``_monitor_vars_learn`` method.

        .. note::
            The input value can be torch.Tensor or dict/list combinations and current policy supports all of them. \
            For the data type that not supported, the main reason is that the corresponding model does not support it. \
            You can implement you own model rather than use the default model. For more information, please raise an \
            issue in GitHub repo and we will continue to follow up.
        """
        if isinstance(data, list):
            data = default_collate(data)
        if self._cuda:
            data = to_device(data, self._device)
        self._learn_model.train()
        with self._timer:
            obs, action = data['obs'], data['action'].squeeze()
            if self._cfg.continuous:
                if self._cfg.learn.tanh_mask:
                    """tanh_mask
                    We mask the action out of range of [tanh(-1),tanh(1)], model will learn information
                    and produce action in [-1,1]. So the action won't always converge to -1 or 1.
                    """
                    mu = self._eval_model.forward(data['obs'])['action']
                    bound = 1 - 2 / (math.exp(2) + 1)  # tanh(1): (e-e**(-1))/(e+e**(-1))
                    mask = mu.ge(-bound) & mu.le(bound)
                    mask_percent = 1 - mask.sum().item() / mu.numel()
                    if mask_percent > 0.8:  # if there is too little data to learn(<80%). So we use all data.
                        loss = self._loss(mu, action.detach())
                    else:
                        loss = self._loss(mu.masked_select(mask), action.masked_select(mask).detach())
                else:
                    mu = self._learn_model.forward(data['obs'])['action']
                    # When we use bco, action is predicted by idm, gradient is not expected.
                    loss = self._loss(mu, action.detach())
            else:
                a_logit = self._learn_model.forward(obs)
                # When we use bco, action is predicted by idm, gradient is not expected.
                loss = self._loss(a_logit['logit'], action.detach())

                if self._cfg.learn.show_accuracy:
                    # Calculate the overall accuracy and the accuracy of each class
                    total_accuracy = (a_logit['action'] == action.view(-1)).float().mean()
                    self.total_accuracy_in_dataset.append(total_accuracy)
                    logging.info(f'the total accuracy in current train mini-batch is: {total_accuracy.item()}')
                    for action_unique in to_list(torch.unique(action)):
                        action_index = (action == action_unique).nonzero(as_tuple=True)[0]
                        action_accuracy = (a_logit['action'][action_index] == action.view(-1)[action_index]
                                           ).float().mean()
                        if math.isnan(action_accuracy):
                            action_accuracy = 0.0
                        self.action_accuracy_in_dataset[action_unique].append(action_accuracy)
                        logging.info(
                            f'the accuracy of action {action_unique} in current train mini-batch is: '
                            f'{action_accuracy.item()}, '
                            f'(nan means the action does not appear in the mini-batch)'
                        )
        forward_time = self._timer.value
        with self._timer:
            self._optimizer.zero_grad()
            loss.backward()
        backward_time = self._timer.value
        with self._timer:
            if self._cfg.multi_gpu:
                self.sync_gradients(self._learn_model)
        sync_time = self._timer.value
        self._optimizer.step()
        cur_lr = [param_group['lr'] for param_group in self._optimizer.param_groups]
        cur_lr = sum(cur_lr) / len(cur_lr)
        return {
            'cur_lr': cur_lr,
            'total_loss': loss.item(),
            'forward_time': forward_time,
            'backward_time': backward_time,
            'sync_time': sync_time,
        }

    def _monitor_vars_learn(self) -> List[str]:
        """
        Overview:
            Return the necessary keys for logging the return dict of ``self._forward_learn``. The logger module, such \
            as text logger, tensorboard logger, will use these keys to save the corresponding data.
        Returns:
            - necessary_keys (:obj:`List[str]`): The list of the necessary keys to be logged.
        """
        return ['cur_lr', 'total_loss', 'forward_time', 'backward_time', 'sync_time']

    def _init_eval(self):
        """
        Overview:
            Initialize the eval mode of policy, including related attributes and modules. For BC, it contains the \
            eval model to greedily select action with argmax q_value mechanism for discrete action space.
            This method will be called in ``__init__`` method if ``eval`` field is in ``enable_field``.

        .. note::
            If you want to set some spacial member variables in ``_init_eval`` method, you'd better name them \
            with prefix ``_eval_`` to avoid conflict with other modes, such as ``self._eval_attr1``.
        """
        if self._cfg.continuous:
            self._eval_model = model_wrap(self._model, wrapper_name='base')
        else:
            self._eval_model = model_wrap(self._model, wrapper_name='argmax_sample')
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
                key of the dict is the same as the input data, i.e. environment id.

        .. note::
            The input value can be torch.Tensor or dict/list combinations and current policy supports all of them. \
            For the data type that not supported, the main reason is that the corresponding model does not support it. \
            You can implement you own model rather than use the default model. For more information, please raise an \
            issue in GitHub repo and we will continue to follow up.
        """
        tensor_input = isinstance(data, torch.Tensor)
        if tensor_input:
            data = default_collate(list(data))
        else:
            data_id = list(data.keys())
            data = default_collate(list(data.values()))
        if self._cuda:
            data = to_device(data, self._device)
        self._eval_model.eval()
        with torch.no_grad():
            output = self._eval_model.forward(data)
        if self._cuda:
            output = to_device(output, 'cpu')
        if tensor_input:
            return output
        else:
            output = default_decollate(output)
            return {i: d for i, d in zip(data_id, output)}

    def _init_collect(self) -> None:
        """
        Overview:
            BC policy uses offline dataset so it does not need to collect data. However, sometimes we need to use the \
            trained BC policy to collect data for other purposes.
        """
        self._unroll_len = self._cfg.collect.unroll_len
        if self._cfg.continuous:
            self._collect_model = model_wrap(
                self._model,
                wrapper_name='action_noise',
                noise_type='gauss',
                noise_kwargs={
                    'mu': 0.0,
                    'sigma': self._cfg.collect.noise_sigma.start
                },
                noise_range=self._cfg.collect.noise_range
            )
        else:
            self._collect_model = model_wrap(self._model, wrapper_name='eps_greedy_sample')
        self._collect_model.reset()

    def _forward_collect(self, data: Dict[int, Any], **kwargs) -> Dict[int, Any]:
        data_id = list(data.keys())
        data = default_collate(list(data.values()))
        if self._cuda:
            data = to_device(data, self._device)
        self._collect_model.eval()
        with torch.no_grad():
            if self._cfg.continuous:
                # output = self._collect_model.forward(data)
                output = self._collect_model.forward(data, **kwargs)
            else:
                output = self._collect_model.forward(data, **kwargs)
        if self._cuda:
            output = to_device(output, 'cpu')
        output = default_decollate(output)
        return {i: d for i, d in zip(data_id, output)}

    def _process_transition(self, obs: Any, policy_output: dict, timestep: namedtuple) -> dict:
        transition = {
            'obs': obs,
            'next_obs': timestep.obs,
            'action': policy_output['action'],
            'reward': timestep.reward,
            'done': timestep.done,
        }
        return EasyDict(transition)

    def _get_train_sample(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        data = get_nstep_return_data(data, 1, 1)
        return get_train_sample(data, self._unroll_len)
