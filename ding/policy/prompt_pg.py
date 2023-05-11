from typing import List, Dict, Any, Tuple, Union
from collections import namedtuple
import torch

from ding.rl_utils import get_gae_with_default_last_value, get_train_sample
from ding.torch_utils import Adam, to_device
from ding.utils import POLICY_REGISTRY, split_data_generator
from ding.utils.data import default_collate, default_decollate
from .base_policy import Policy
from .common_utils import default_preprocess_learn


@POLICY_REGISTRY.register('prompt_pg')
class PromptPGPolicy(Policy):
    r"""
    Overview:
        Policy class of Prompt Policy Gradient (PromptPG) algorithm.
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
        return 'nlp_pretrained_model', ['ding.model.template.nlp_pretrained_model']

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
            - data (:obj:`dict`): Dict type data, including at least ['obs', 'action', 'reward', 'next_obs','adv']
        Returns:
            - info_dict (:obj:`Dict[str, Any]`): Including current lr and loss.
        """
        print(data)
        self._model.train()

        return_infos = []
        for batch in split_data_generator(data, self._cfg.learn.batch_size, shuffle=True):
            # forward
            train_samples, cand_samples = batch["train_sample"], batch["candidate_samples"]
            output = self._learn_model.forward(train_samples, cand_samples)
            return_ = batch['return']

            # calculate PG loss
            for ii in range(self._cfg.shot_number):
                log_prob = output['dist'].log_prob(batch['action'][ii])
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
        self._unroll_len = self._cfg.collect.unroll_len
        self._gamma = self._cfg.collect.discount_factor

    def _forward_collect(self, data: dict) -> dict:
        data_id = list(data.keys())
        data = default_collate(list(data.values()))
        self._model.eval()
        with torch.no_grad():
            output = self._model.forward(data['train_sample'], data['candidate_samples'])
            act = []
            mask = torch.zeros_like(output['logit'])
            for ii in range(self._cfg.shot_number):
                dist = torch.distributions.Categorical(logits=output['logit'] + mask)
                actions = dist.sample()
                act.append(actions)
                for jj in range(actions.shape[0]):
                    mask[jj][actions[jj]] = -1e30
            act = torch.stack(act, dim=0)
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
        assert data[-1]['done'] is True, "PG needs a complete epsiode"

        if self._cfg.learn.ignore_done:
            raise NotImplementedError

        R = 0.
        for i in reversed(range(len(data))):
            R = self._gamma * R + data[i]['reward']
            data[i]['return'] = R
        return get_train_sample(data, self._unroll_len)

    def _init_eval(self) -> None:
        pass

    def _forward_eval(self, data: dict) -> dict:
        data_id = list(data.keys())
        data = default_collate(list(data.values()))
        self._model.eval()
        with torch.no_grad():
            output = self._model.forward(data['train_sample'], data['candidate_samples'])
            act = []
            mask = torch.zeros_like(output['logit'])
            for ii in range(self._cfg.shot_number):
                actions = torch.argmax(output['logit'] + mask, dim=-1)
                act.append(actions)
                for jj in range(actions.shape[0]):
                    mask[jj][actions[jj]] = -1e30
            act = torch.stack(act, dim=0)
        if self._cuda:
            output = to_device(output, 'cpu')
        output = default_decollate(output)
        return {i: d for i, d in zip(data_id, output)}

    def _monitor_vars_learn(self) -> List[str]:
        return super()._monitor_vars_learn() + ['policy_loss', 'entropy_loss', 'return_abs_max', 'grad_norm']
