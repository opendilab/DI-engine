from typing import List, Dict, Any, Tuple, Union
from collections import namedtuple
import torch

from ding.rl_utils import get_train_sample
from ding.torch_utils import Adam, to_device
from ding.utils import POLICY_REGISTRY, split_data_generator
from ding.utils.data import default_collate, default_decollate
from .base_policy import Policy
from ..model import model_wrap


@POLICY_REGISTRY.register('prompt_pg')
class PromptPGPolicy(Policy):
    r"""
    Overview:
        Policy class of Prompt Policy Gradient (PromptPG) algorithm.
        Link of the original paper: https://arxiv.org/abs/2209.14610
    """
    config = dict(
        # (string) RL policy register name (refer to function "register_policy").
        type='prompt_pg',
        # (bool) whether to use cuda for network.
        cuda=True,
        # (bool) whether use on-policy training pipeline(behaviour policy and training policy are the same)
        on_policy=True,  # for pg strictly on policy algorithm, this line should not be modified by users
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
            discount_factor=0,
            collector=dict(get_train_sample=True),
        ),
        eval=dict(),
    )

    def default_model(self) -> Tuple[str, List[str]]:
        return 'language_transformer', ['ding.model.template.language_transformer']

    def _init_learn(self) -> None:
        r"""
        Overview:
            Learn mode init method. Called by ``self.__init__``.
            Init the optimizer, algorithm config, main and target models.
        """
        # Optimizer
        self._optimizer = Adam(self._model.parameters(), lr=self._cfg.learn.learning_rate)

        self._entropy_weight = self._cfg.learn.entropy_weight
        self._grad_norm = self._cfg.learn.grad_norm
        self._learn_model = self._model  # for compatibility

    def _forward_learn(self, data: dict) -> Dict[str, Any]:
        r"""
        Overview:
            Forward and backward function of learn mode.
        Arguments:
            - data (:obj:`dict`): Dict type data, including at least ['obs', 'action', 'reward']
        Returns:
            - info_dict (:obj:`Dict[str, Any]`): Including current lr and loss.
        """
        self._model.train()

        return_infos = []
        for i in range(0, len(data), self._cfg.learn.batch_size):
            batch = default_collate(data[i:i + self._cfg.learn.batch_size])
            if self._cuda:
                batch = to_device(batch, self._device)

            # Prepare train_sample (the question to be answered) and the candidate_samples (the prompts to be selected)
            train_samples, cand_samples = batch["obs"]["train_sample"], batch["obs"]["candidate_samples"]
            for ii in range(len(cand_samples)):
                cand_samples[ii] = cand_samples[ii][0]
            output = self._learn_model.forward(train_samples, cand_samples)
            return_ = batch['return']

            # calculate PG loss
            real_act = batch['action']  # shape: (B, shot_number)
            # Calculate loss.
            total_policy_loss, total_entropy_loss = 0, 0
            for ii in range(self._cfg.shot_number):
                log_prob = output['dist'].log_prob(real_act[:, ii])
                policy_loss = -(log_prob * return_).mean()
                total_policy_loss += policy_loss
            total_entropy_loss += -self._cfg.learn.entropy_weight * output['dist'].entropy().mean()
            total_loss = total_entropy_loss + total_policy_loss

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
                'policy_loss': total_policy_loss.item(),
                'entropy_loss': total_entropy_loss.item(),
                'return_abs_max': return_.abs().max().item(),
                'grad_norm': grad_norm,
            }
            return_infos.append(return_info)
        return return_infos

    def _init_collect(self) -> None:
        self._unroll_len = self._cfg.collect.unroll_len
        self._gamma = self._cfg.collect.discount_factor
        self._collect_model = model_wrap(self._model, wrapper_name='combination_multinomial_sample')

    def _forward_collect(self, data: dict) -> dict:
        data_id = list(data.keys())
        data = default_collate(list(data.values()))
        self._model.eval()
        with torch.no_grad():
            # Prepare train_sample (the question to be answered) and the candidate_samples (the prompts to be selected)
            for ii in range(len(data['candidate_samples'])):
                data['candidate_samples'][ii] = data['candidate_samples'][ii][0]
            output = self._collect_model.forward(self._cfg.shot_number, data['train_sample'], data['candidate_samples'])
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
                (here 'obs' indicates obs after env step).
        Returns:
            - transition (:obj:`dict`): Dict type transition data.
        """
        return {
            'obs': obs,
            'action': model_output['action'],
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
        return super()._monitor_vars_learn() + ['policy_loss', 'entropy_loss', 'return_abs_max', 'grad_norm']
