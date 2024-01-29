from typing import List, Dict, Any, Tuple, Union
from collections import namedtuple
import torch
import treetensor as ttorch

from ding.rl_utils import get_gae_with_default_last_value, get_train_sample
from ding.torch_utils import Adam, to_device
from ding.utils import POLICY_REGISTRY, split_data_generator
from ding.utils.data import default_collate, default_decollate
from .base_policy import Policy
from .common_utils import default_preprocess_learn


@POLICY_REGISTRY.register('pg')
class PGPolicy(Policy):
    """
    Overview:
        Policy class of Policy Gradient (REINFORCE) algorithm. Paper link: \
        https://proceedings.neurips.cc/paper_files/paper/1999/file/464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf
    """
    config = dict(
        # (string) RL policy register name (refer to function "register_policy").
        type='pg',
        # (bool) whether to use cuda for network.
        cuda=False,
        # (bool) whether use on-policy training pipeline(behaviour policy and training policy are the same)
        on_policy=True,  # for pg strictly on policy algorithm, this line should not be modified by users
        # (str) action space type: ['discrete', 'continuous']
        action_space='discrete',
        # (bool) whether to use deterministic action for evaluation.
        deterministic_eval=True,
        learn=dict(

            # (int) the number of samples for one update.
            batch_size=64,
            # (float) the step size of one gradient descend.
            learning_rate=0.001,
            # ==============================================================
            # The following configs is algorithm-specific
            # ==============================================================
            # (float) loss weight of the entropy regularization, the weight of policy network is set to 1
            entropy_weight=0.01,
            # (float) max grad norm value.
            grad_norm=5,
            # (bool) whether to ignore done signal for non-termination env.
            ignore_done=False,
        ),
        collect=dict(
            # (int) collect n_sample data, train model n_iteration times
            # n_episode=8,
            # (int) trajectory unroll length
            unroll_len=1,
            # ==============================================================
            # The following configs is algorithm-specific
            # ==============================================================
            # (float) discount factor for future reward, defaults int [0, 1]
            discount_factor=0.99,
            collector=dict(get_train_sample=True),
        ),
        eval=dict(),
    )

    def default_model(self) -> Tuple[str, List[str]]:
        """
        Overview:
            Return this algorithm default neural network model setting for demonstration. ``__init__`` method will \
            automatically call this method to get the default model setting and create model.
        Returns:
            - model_info (:obj:`Tuple[str, List[str]]`): The registered model name and model's import_names.
        """
        return 'pg', ['ding.model.template.pg']

    def _init_learn(self) -> None:
        """
        Overview:
            Initialize the learn mode of policy, including related attributes and modules. For PG, it mainly \
            contains optimizer, algorithm-specific arguments such as entropy weight and grad norm. This method \
            also executes some special network initializations.
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
        # Optimizer
        self._optimizer = Adam(self._model.parameters(), lr=self._cfg.learn.learning_rate)

        self._entropy_weight = self._cfg.learn.entropy_weight
        self._grad_norm = self._cfg.learn.grad_norm
        self._learn_model = self._model  # for compatibility

    def _forward_learn(self, data: List[Dict[int, Any]]) -> Dict[str, Any]:
        """
        Overview:
            Policy forward function of learn mode (training policy and updating parameters). Forward means \
            that the policy inputs some training batch data from the replay buffer and then returns the output \
            result, including various training information such as loss, clipfrac, approx_kl.
        Arguments:
            - data (:obj:`List[Dict[int, Any]]`): The input data used for policy forward, including the latest \
                collected training samples for on-policy algorithms like PG. For each element in list, the key of the \
                dict is the name of data items and the value is the corresponding data. Usually, the value is \
                torch.Tensor or np.ndarray or there dict/list combinations. In the ``_forward_learn`` method, data \
                often need to first be stacked in the batch dimension by some utility functions such as \
                ``default_preprocess_learn``. \
                For PG, each element in list is a dict containing at least the following keys: ``obs``, ``action``, \
                ``return``.
        Returns:
            - return_infos (:obj:`List[Dict[str, Any]]`): The information list that indicated training result, each \
                training iteration contains append a information dict into the final list. The list will be precessed \
                and recorded in text log and tensorboard. The value of the dict must be python scalar or a list of \
                scalars. For the detailed definition of the dict, refer to the code of ``_monitor_vars_learn`` method.

        .. note::
            The input value can be torch.Tensor or dict/list combinations and current policy supports all of them. \
            For the data type that not supported, the main reason is that the corresponding model does not support it. \
            You can implement you own model rather than use the default model. For more information, please raise an \
            issue in GitHub repo and we will continue to follow up.
        """
        data = default_preprocess_learn(data, ignore_done=self._cfg.learn.ignore_done, use_nstep=False)
        if self._cuda:
            data = to_device(data, self._device)
        self._model.train()

        return_infos = []
        for batch in split_data_generator(data, self._cfg.learn.batch_size, shuffle=True):
            # forward
            output = self._learn_model.forward(batch['obs'])
            return_ = batch['return']
            dist = output['dist']
            # calculate PG loss
            log_prob = dist.log_prob(batch['action'])
            policy_loss = -(log_prob * return_).mean()
            entropy_loss = -self._cfg.learn.entropy_weight * dist.entropy().mean()
            total_loss = policy_loss + entropy_loss

            # update
            self._optimizer.zero_grad()
            total_loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(
                list(self._learn_model.parameters()),
                max_norm=self._grad_norm,
            )
            self._optimizer.step()

            # only record last updates information in logger
            return_info = {
                'cur_lr': self._optimizer.param_groups[0]['lr'],
                'total_loss': total_loss.item(),
                'policy_loss': policy_loss.item(),
                'entropy_loss': entropy_loss.item(),
                'return_abs_max': return_.abs().max().item(),
                'grad_norm': grad_norm,
            }
            return_infos.append(return_info)
        return return_infos

    def _init_collect(self) -> None:
        """
        Overview:
            Initialize the collect mode of policy, including related attributes and modules. For PPG, it contains \
            algorithm-specific arguments such as unroll_len and gamma.
            This method will be called in ``__init__`` method if ``collect`` field is in ``enable_field``.

        .. note::
            If you want to set some spacial member variables in ``_init_collect`` method, you'd better name them \
            with prefix ``_collect_`` to avoid conflict with other modes, such as ``self._collect_attr1``.
        """
        self._unroll_len = self._cfg.collect.unroll_len
        self._gamma = self._cfg.collect.discount_factor

    def _forward_collect(self, data: Dict[int, Any]) -> dict:
        """
        Overview:
            Policy forward function of collect mode (collecting training data by interacting with envs). Forward means \
            that the policy gets some necessary data (mainly observation) from the envs and then returns the output \
            data, such as the action to interact with the envs.
        Arguments:
            - data (:obj:`Dict[int, Any]`): The input data used for policy forward, including at least the obs. The \
                key of the dict is environment id and the value is the corresponding data of the env.
        Returns:
            - output (:obj:`Dict[int, Any]`): The output data of policy forward, including at least the action and \
                other necessary data (action logit) for learn mode defined in ``self._process_transition`` \
                method. The key of the dict is the same as the input data, i.e. environment id.

        .. tip::
            If you want to add more tricks on this policy, like temperature factor in multinomial sample, you can pass \
            related data as extra keyword arguments of this method.

        .. note::
            The input value can be torch.Tensor or dict/list combinations and current policy supports all of them. \
            For the data type that not supported, the main reason is that the corresponding model does not support it. \
            You can implement you own model rather than use the default model. For more information, please raise an \
            issue in GitHub repo and we will continue to follow up.
        """
        data_id = list(data.keys())
        data = default_collate(list(data.values()))
        if self._cuda:
            data = to_device(data, self._device)
        self._model.eval()
        with torch.no_grad():
            output = self._model.forward(data)
            output['action'] = output['dist'].sample()
        if self._cuda:
            output = to_device(output, 'cpu')
        output = default_decollate(output)
        return {i: d for i, d in zip(data_id, output)}

    def _process_transition(self, obs: Any, model_output: Dict[str, torch.Tensor], timestep: namedtuple) -> dict:
        """
        Overview:
            Process and pack one timestep transition data into a dict, which can be directly used for training and \
            saved in replay buffer. For PG, it contains obs, action, reward, done.
        Arguments:
            - obs (:obj:`torch.Tensor`): The env observation of current timestep, such as stacked 2D image in Atari.
            - model_output (:obj:`Dict[str, torch.Tensor]`): The output of the policy network with the observation \
                as input. For PG, it contains the action.
            - timestep (:obj:`namedtuple`): The execution result namedtuple returned by the environment step method, \
                except all the elements have been transformed into tensor data. Usually, it contains the next obs, \
                reward, done, info, etc.
        Returns:
            - transition (:obj:`Dict[str, torch.Tensor]`): The processed transition data of the current timestep.
        """
        return {
            'obs': obs,
            'action': model_output['action'],
            'reward': timestep.reward,
            'done': timestep.done,
        }

    def _get_train_sample(self, data: List[Dict[str, Any]]) -> Union[None, List[Any]]:
        """
        Overview:
            For a given entire episode data (a list of transition), process it into a list of sample that \
            can be used for training directly. In PG, a train sample is a processed transition with new computed \
            ``return`` field. This method is usually used in collectors to execute necessary \
            RL data preprocessing before training, which can help learner amortize revelant time consumption. \
            In addition, you can also implement this method as an identity function and do the data processing \
            in ``self._forward_learn`` method.
        Arguments:
            - data (:obj:`List[Dict[str, Any]`): The episode data (a list of transition), each element is \
                the same format as the return value of ``self._process_transition`` method. Note that PG needs \
                a complete epsiode
        Returns:
            - samples (:obj:`List[Dict[str, Any]]`): The processed train samples, each element is the similar format \
                as input transitions, but may contain more data for training, such as discounted episode return.
        """
        assert data[-1]['done'], "PG needs a complete epsiode"

        if self._cfg.learn.ignore_done:
            raise NotImplementedError

        R = 0.
        if isinstance(data, list):
            for i in reversed(range(len(data))):
                R = self._gamma * R + data[i]['reward']
                data[i]['return'] = R
            return get_train_sample(data, self._unroll_len)
        elif isinstance(data, ttorch.Tensor):
            data_size = data['done'].shape[0]
            data['return'] = ttorch.torch.zeros(data_size)
            for i in reversed(range(data_size)):
                R = self._gamma * R + data['reward'][i]
                data['return'][i] = R
            return get_train_sample(data, self._unroll_len)
        else:
            raise ValueError

    def _init_eval(self) -> None:
        pass

    def _forward_eval(self, data: Dict[int, Any]) -> dict:
        """
        Overview:
            Policy forward function of eval mode (evaluation policy performance by interacting with envs). Forward \
            means that the policy gets some necessary data (mainly observation) from the envs and then returns the \
            action to interact with the envs. ``_forward_eval`` in PG often uses deterministic sample method to get \
            actions while ``_forward_collect`` usually uses stochastic sample method for balance exploration and \
            exploitation.
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

        .. note::
            For more detailed examples, please refer to our unittest for PGPGPolicy: ``ding.policy.tests.test_pg``.
        """
        data_id = list(data.keys())
        data = default_collate(list(data.values()))
        if self._cuda:
            data = to_device(data, self._device)
        self._model.eval()
        with torch.no_grad():
            output = self._model.forward(data)
            if self._cfg.deterministic_eval:
                if self._cfg.action_space == 'discrete':
                    output['action'] = output['logit'].argmax(dim=-1)
                elif self._cfg.action_space == 'continuous':
                    output['action'] = output['logit']['mu']
                else:
                    raise KeyError("invalid action_space: {}".format(self._cfg.action_space))
            else:
                output['action'] = output['dist'].sample()
        if self._cuda:
            output = to_device(output, 'cpu')
        output = default_decollate(output)
        return {i: d for i, d in zip(data_id, output)}

    def _monitor_vars_learn(self) -> List[str]:
        """
        Overview:
            Return the necessary keys for logging the return dict of ``self._forward_learn``. The logger module, such \
            as text logger, tensorboard logger, will use these keys to save the corresponding data.
        Returns:
            - necessary_keys (:obj:`List[str]`): The list of the necessary keys to be logged.
        """
        return super()._monitor_vars_learn() + ['policy_loss', 'entropy_loss', 'return_abs_max', 'grad_norm']
