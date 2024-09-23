from collections import namedtuple
from typing import List, Dict, Any, Tuple, Union

import torch

from ding.model import model_wrap
from ding.rl_utils import get_train_sample
from ding.torch_utils import Adam, to_device
from ding.utils import POLICY_REGISTRY
from ding.utils.data import default_collate, default_decollate
from .base_policy import Policy


@POLICY_REGISTRY.register('prompt_awr')
class PromptAWRPolicy(Policy):
    """
    Overview:
        Policy class of AWR (Advantage Weighted Regression) algorithm, proposed in https://arxiv.org/abs/1910.00177.
        Especially, this policy is designed for training a language model policy.
        In this policy, the environment's observation includes the current context, a list of optional actions
        (strings). The final output of the policy is a set of optional actions with a size of ``shot_number``.
    """
    config = dict(
        # (str) Name of the registered RL policy (refer to the "register_policy" function).
        type='prompt_awr',
        # (bool) Flag to enable CUDA for model computation.
        cuda=False,
        # (bool) Flag for using on-policy training (training policy is the same as the behavior policy).
        on_policy=False,
        # (bool) Flag for enabling priority experience replay. Must be False when priority_IS_weight is False.
        priority=False,
        # (bool) Flag for using Importance Sampling weights to correct updates. Requires `priority` to be True.
        priority_IS_weight=False,
        # (str) Type of action space used in the policy, with valid options ['discrete', 'continuous'].
        action_space='discrete',
        # (int) The number of actions that can be done simultaneously in one timestep.
        shot_number=1,
        # learn_mode configuration
        learn=dict(
            # (int) Number of updates per data collection. A2C requires this to be set to 1.
            update_per_collect=1,
            # (int) Batch size for learning.
            batch_size=64,
            # (float) Learning rate for optimizer.
            learning_rate=0.001,
            # (Tuple[float, float]) Coefficients used for computing running averages of gradient and its square.
            betas=(0.9, 0.999),
            # (float) Term added to the denominator to improve numerical stability in optimizer.
            eps=1e-8,
            # (float) Maximum norm for gradients.
            grad_norm=0.5,
            # (float) Scaling factor for value network loss relative to policy network loss.
            value_weight=0.5,
            # (float) Coefficient that controls the exp scale in awr algorithm.
            beta=1.0,
            # (float) Weight of entropy regularization in the loss function.
            entropy_weight=0.001,
            # (Tuple[float, float]) The range of adv. Value that exceeds this range will be clipped.
            adv_range=(-0.5, 0.5),
            # (bool) If set to True, the 'done' signals that indicate the end of an episode due to environment time
            # limits are disregarded. By default, this is set to False. This setting is particularly useful for tasks
            # that have a predetermined episode length, such as HalfCheetah and various other MuJoCo environments,
            # where the maximum length is capped at 1000 steps. When enabled, any 'done' signal triggered by reaching
            # the maximum episode steps will be overridden to 'False'. This ensures the accurate calculation of the
            # Temporal Difference (TD) error, using the formula `gamma * (1 - done) * next_v + reward`,
            # even when the episode surpasses the predefined step limit.
            ignore_done=False,
        ),
        # collect_mode configuration
        collect=dict(
            # (int) The length of rollout for data collection.
            unroll_len=1,
            # (float) Discount factor for calculating future rewards, typically in the range [0, 1].
            discount_factor=0.9,
            # (float) Trade-off parameter for balancing TD-error and Monte Carlo error in GAE.
            gae_lambda=0.95,
        ),
        # eval_mode configuration (kept empty for compatibility purposes)
        eval=dict(),
    )

    def default_model(self) -> Tuple[str, List[str]]:
        """
        Overview:
            Returns the default model configuration used by the AWR algorithm. ``__init__`` method will \
            automatically call this method to get the default model setting and create model.

        Returns:
            - model_info (:obj:`Tuple[str, List[str]]`): \
                Tuple containing the registered model name and model's import_names.
        """
        return 'language_transformer', ['ding.model.template.language_transformer']

    def _init_learn(self) -> None:
        """
        Overview:
            Initialize the learn mode of policy, including related attributes and modules. For AWR, it mainly \
            contains optimizer, algorithm-specific arguments such as value_weight, entropy_weight, adv_norm
            and grad_norm, and main model. \
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
        assert self._cfg.action_space == "discrete"
        # Optimizer
        self._optimizer = Adam(
            self._model.parameters(),
            lr=self._cfg.learn.learning_rate,
            betas=self._cfg.learn.betas,
            eps=self._cfg.learn.eps
        )

        # Algorithm config
        self._priority = self._cfg.priority
        self._priority_IS_weight = self._cfg.priority_IS_weight
        self._value_weight = self._cfg.learn.value_weight
        self._entropy_weight = self._cfg.learn.entropy_weight
        self._adv_norm = self._cfg.learn.adv_norm
        self._grad_norm = self._cfg.learn.grad_norm

        # Main and target models
        self._learn_model = self._model

    def _forward_learn(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Data preprocessing operations, such as stack data, cpu to cuda device
        self._learn_model.train()

        for i in range(0, len(data), self._cfg.learn.batch_size):
            batch = default_collate(data[i:i + self._cfg.learn.batch_size])
            if self._cuda:
                batch = to_device(batch, self._device)

            # Prepare train_sample (the question to be answered) and the candidate_samples (the prompts to be selected)
            train_samples, cand_samples = batch["obs"]["train_sample"], batch["obs"]["candidate_samples"]
            for cand_n in range(len(cand_samples)):
                cand_samples[cand_n] = cand_samples[cand_n][0]
            output = self._learn_model.forward(train_samples, cand_samples, mode='compute_actor_critic')
            return_ = batch['return']

            # Calculate AWR loss
            real_act = batch['action']

            # Ensure the shape of real_act is: (B, shot_number)
            if len(real_act.shape) == 1:
                real_act = real_act.unsqueeze(-1)

            # Calculate different parts of loss.
            total_policy_loss, total_entropy_loss, total_value_loss = 0, 0, 0
            for shot_n in range(self._cfg.shot_number):
                log_prob = output['dist'].log_prob(real_act[:, shot_n])
                # Clamp the adv for better stability.
                adv = torch.clamp(
                    return_ - batch['value'], min=self._cfg.learn.norm_range[0], max=self._cfg.learn.norm_range[1]
                )
                # The policy loss for AWR algorithm.
                policy_loss = -(log_prob * torch.exp(adv / self._cfg.learn.beta)).mean()
                total_policy_loss += policy_loss
            # The value loss for AWR algorithm.
            value_loss = ((return_ - output['value']) ** 2).mean()
            total_value_loss += value_loss
            # The entropy loss for AWR algorithm.
            total_entropy_loss += -self._cfg.learn.entropy_weight * output['dist'].entropy().mean()
            total_loss = total_entropy_loss + total_policy_loss + total_value_loss

            self._optimizer.zero_grad()
            total_loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(
                list(self._learn_model.parameters()),
                max_norm=self._grad_norm,
            )
            self._optimizer.step()

        return {
            'cur_lr': self._optimizer.param_groups[0]['lr'],
            'total_loss': total_loss.item(),
            'policy_loss': total_policy_loss.item(),
            'entropy_loss': total_entropy_loss.item(),
            'value_loss': total_value_loss.item(),
            'return_abs_max': return_.abs().max().item(),
            'grad_norm': grad_norm,
        }

    def _init_collect(self) -> None:
        self._unroll_len = self._cfg.collect.unroll_len
        self._gamma = self._cfg.collect.discount_factor
        self._collect_model = model_wrap(self._model, wrapper_name='combination_multinomial_sample')

    def _forward_collect(self, data: Dict[int, Any]) -> Dict[int, Any]:
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
                other necessary data for learn mode defined in ``self._process_transition`` method. The key of the \
                dict is the same as the input data, i.e. environment id.
        """
        data_id = list(data.keys())
        data = default_collate(list(data.values()))
        self._model.eval()
        with torch.no_grad():
            # Prepare train_sample (the question to be answered) and the candidate_samples (the prompts to be selected)
            for ii in range(len(data['candidate_samples'])):
                data['candidate_samples'][ii] = data['candidate_samples'][ii][0]
            output = self._collect_model.forward(
                self._cfg.shot_number, data['train_sample'], data['candidate_samples'], mode="compute_actor_critic"
            )
        if self._cuda:
            output = to_device(output, 'cpu')
        output = default_decollate(output)
        return {i: d for i, d in zip(data_id, output)}

    def _process_transition(self, obs: Any, policy_output: Dict[str, torch.Tensor],
                            timestep: namedtuple) -> Dict[str, torch.Tensor]:
        return {
            'obs': obs,
            'action': policy_output['action'],
            'value': policy_output['value'],
            'reward': timestep.reward,
            'done': timestep.done,
        }

    def _get_train_sample(self, data: list) -> Union[None, List[Any]]:
        r"""
        Overview:
            Get the trajectory and the n step return data, then sample from the n_step return data
        Arguments:
            - data (:obj:`list`): The trajectory's buffer list
        Returns:
            - samples (:obj:`dict`): The training samples generated
        """
        if self._cfg.learn.ignore_done:
            raise NotImplementedError

        R = 0.
        for i in reversed(range(len(data))):
            R = self._gamma * R + data[i]['reward']
            data[i]['return'] = R
        return get_train_sample(data, self._unroll_len)

    def _init_eval(self) -> None:
        self._eval_model = model_wrap(self._model, wrapper_name='combination_argmax_sample')

    def _forward_eval(self, data: dict) -> dict:
        data_id = list(data.keys())
        data = default_collate(list(data.values()))
        self._model.eval()
        with torch.no_grad():
            # Prepare train_sample (the question to be answered) and the candidate_samples (the prompts to be selected)
            for ii in range(len(data['candidate_samples'])):
                data['candidate_samples'][ii] = data['candidate_samples'][ii][0]
            output = self._eval_model.forward(self._cfg.shot_number, data['train_sample'], data['candidate_samples'])
        if self._cuda:
            output = to_device(output, 'cpu')
        output = default_decollate(output)
        return {i: d for i, d in zip(data_id, output)}

    def _monitor_vars_learn(self) -> List[str]:
        return super()._monitor_vars_learn() + \
               ['policy_loss', 'entropy_loss', 'return_abs_max', 'grad_norm', 'value_loss']
