import math
import itertools
import numpy as np
import torch
import torch.nn.functional as F
import copy
from collections import namedtuple, deque
from typing import List, Dict, Any, Tuple, Union, Optional
from torch.distributions.categorical import Categorical

from nervex.torch_utils import Adam, to_device
from nervex.data import default_collate, default_decollate
from nervex.rl_utils import Adder
from nervex.armor import model_wrap
from nervex.model import FCDiscreteNet, SQNDiscreteNet
from nervex.utils import POLICY_REGISTRY
from .base_policy import Policy
from .common_utils import default_preprocess_learn


class SQNModel(torch.nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super(SQNModel, self).__init__()
        self.q0 = SQNDiscreteNet(*args, **kwargs)
        self.q1 = SQNDiscreteNet(*args, **kwargs)

    def forward(self, data: dict) -> dict:
        output0 = self.q0(data)
        output1 = self.q1(data)
        return {
            'q_value': [output0['logit'], output1['logit']],
            'logit': output0['logit'],
        }


@POLICY_REGISTRY.register('sqn')
class SQNPolicy(Policy):
    r"""
    Overview:
        Policy class of SQN algorithm (arxiv: 1912.10891).
    """

    def _init_learn(self) -> None:
        r"""
        Overview:
            Learn mode init method. Called by ``self.__init__``.
            Init q, value and policy's optimizers, algorithm config, main and target armors.
        """
        # Optimizers
        self._optimizer_q = Adam(
            self._model.parameters(), lr=self._cfg.learn.learning_rate_q, weight_decay=self._cfg.learn.weight_decay
        )

        # Algorithm config
        algo_cfg = self._cfg.learn.algo
        self._algo_cfg_learn = algo_cfg
        self._gamma = algo_cfg.discount_factor
        self._action_dim = self._cfg.model.action_dim
        if isinstance(self._action_dim, int):
            self._target_entropy = algo_cfg.get('target_entropy', self._action_dim / 10)
        else:
            self._target_entropy = algo_cfg.get('target_entropy', 0.2)

        self._log_alpha = torch.FloatTensor([math.log(algo_cfg.alpha)]).to(self._device).requires_grad_(True)
        self._optimizer_alpha = torch.optim.Adam([self._log_alpha], lr=self._cfg.learn.learning_rate_alpha)

        # Main and target armors
        self._target_model = copy.deepcopy(self._model)
        self._target_model = model_wrap(
            self._target_model,
            wrapper_name='target',
            update_type='momentum',
            update_kwargs={'theta': algo_cfg.target_theta}
        )
        self._learn_model = model_wrap(self._model, wrapper_name='base')
        self._learn_model.reset()
        self._target_model.reset()

        self._forward_learn_cnt = 0

    def q_1step_td_loss(self, td_data: dict) -> torch.tensor:
        q_value = td_data["q_value"]
        target_q_value = td_data["target_q_value"]
        action = td_data.get('action')
        done = td_data.get('done')
        reward = td_data.get('reward')
        q0 = q_value[0]
        q1 = q_value[1]
        batch_range = torch.arange(action.shape[0])
        q0_a = q0[batch_range, action]
        q1_a = q1[batch_range, action]
        # Target
        with torch.no_grad():
            q0_targ = target_q_value[0]
            q1_targ = target_q_value[1]
            q_targ = torch.min(q0_targ, q1_targ)
            # discrete policy
            alpha = torch.exp(self._log_alpha.clone())
            # TODO use q_targ or q0 for pi
            log_pi = F.log_softmax(q_targ / alpha, dim=-1)
            pi = torch.exp(log_pi)
            # v = \sum_a \pi(a | s) (Q(s, a) - \alpha \log(\pi(a|s)))
            target_v_value = (pi * (q_targ - alpha * log_pi)).sum(axis=-1)
            # q = r + \gamma v
            q_backup = reward + (1 - done) * self._gamma * target_v_value
            # alpha_loss
            entropy = (-pi * log_pi).sum(axis=-1)
            expect_entropy = (pi * self._target_entropy).sum(axis=-1)

        # Q loss
        q0_loss = F.mse_loss(q0_a, q_backup)
        q1_loss = F.mse_loss(q1_a, q_backup)
        total_q_loss = q0_loss + q1_loss
        # alpha loss
        alpha_loss = self._log_alpha * (entropy - expect_entropy).mean()
        return total_q_loss, alpha_loss, entropy

    def _forward_learn(self, data: dict) -> Dict[str, Any]:
        """
        Overview:
            Forward and backward function of learn mode.
        Arguments:
            - data (:obj:`dict`): Dict type data, including at least ['obs', 'action', 'reward', 'next_obs', 'done',\
                'weight']
        Returns:
            - info_dict (:obj:`Dict[str, Any]`): Learn info, including current lr and loss.
        """
        data = default_preprocess_learn(
            data,
            use_priority=self._cfg.get('use_priority', False),
            ignore_done=self._cfg.learn.get('ignore_done', False),
            use_nstep=False
        )
        if self._use_cuda:
            data = to_device(data, self._device)

        self._learn_model.train()
        self._target_model.train()
        obs = data.get('obs')
        next_obs = data.get('next_obs')
        reward = data.get('reward')
        action = data.get('action')
        done = data.get('done')
        # Q-function
        q_value = self._learn_model.forward({'obs': obs})['q_value']
        target_q_value = self._target_model.forward({'obs': next_obs})['q_value']  # TODO:check grad

        num_s_env = 1 if isinstance(self._action_dim, int) else len(self._action_dim)  # num of seperate env

        for s_env_id in range(num_s_env):
            if isinstance(self._action_dim, int):
                td_data = {
                    "q_value": q_value,
                    "target_q_value": target_q_value,
                    "obs": obs,
                    "next_obs": next_obs,
                    "reward": reward,
                    "action": action,
                    "done": done
                }
            else:
                td_data = {
                    "q_value": [q_value[0][s_env_id], q_value[1][s_env_id]],
                    "target_q_value": [target_q_value[0][s_env_id], target_q_value[1][s_env_id]],
                    "obs": obs,
                    "next_obs": next_obs,
                    "reward": reward,
                    "action": action[s_env_id],
                    "done": done
                }
            total_q_loss, alpha_loss, entropy = self.q_1step_td_loss(td_data)
            if s_env_id == 0:
                a_total_q_loss, a_alpha_loss, a_entropy = total_q_loss, alpha_loss, entropy  # accumulate
            else:  # running average, accumulate loss
                a_total_q_loss += total_q_loss / (num_s_env + 1e-6)
                a_alpha_loss += alpha_loss / (num_s_env + 1e-6)
                a_entropy += entropy / (num_s_env + 1e-6)

        self._optimizer_q.zero_grad()
        a_total_q_loss.backward()
        self._optimizer_q.step()

        self._optimizer_alpha.zero_grad()
        a_alpha_loss.backward()
        self._optimizer_alpha.step()

        # target update
        self._target_model.update(self._learn_model.state_dict())
        self._forward_learn_cnt += 1
        # some useful info
        return {
            '[histogram]action_distribution': np.stack([a.cpu().numpy() for a in data['action']]).flatten(),
            'q_loss': a_total_q_loss.item(),
            'alpha_loss': a_alpha_loss.item(),
            'entropy': a_entropy.mean().item(),
            'alpha': math.exp(self._log_alpha.item()),
            'q_value': np.mean([x.cpu().detach().numpy() for x in itertools.chain(*q_value)], dtype=float),
        }

    def _state_dict_learn(self) -> Dict[str, Any]:
        return {
            'model': self._learn_model.state_dict(),
            'optimizer_q': self._optimizer_q.state_dict(),
            'optimizer_alpha': self._optimizer_alpha.state_dict(),
        }

    def _load_state_dict_learn(self, state_dict: Dict[str, Any]) -> None:
        self._learn_model.load_state_dict(state_dict['model'])
        self._optimizer_q.load_state_dict(state_dict['optimizer_q'])
        self._optimizer_alpha.load_state_dict(state_dict['optimizer_alpha'])

    def _init_collect(self) -> None:
        r"""
        Overview:
            Collect mode init method. Called by ``self.__init__``.
            Init traj and unroll length, adder, collect armor.
            Use action noise for exploration.
        """
        self._unroll_len = self._cfg.collect.unroll_len
        self._adder = Adder(self._use_cuda, self._unroll_len)
        self._collect_model = model_wrap(self._model, wrapper_name='base')
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
        if self._use_cuda:
            data = to_device(data, self._device)
        self._collect_model.eval()
        with torch.no_grad():
            # start with random action for better exploration
            output = self._collect_model.forward(data)
        _decay = self._cfg.other.eps.decay
        _act_p = 1 / \
            (_decay - self._forward_learn_cnt) if self._forward_learn_cnt < _decay - 1000 else 0.999

        if np.random.random(1) < _act_p:
            if isinstance(self._action_dim, int):
                logits = output['logit'] / math.exp(self._log_alpha.item())
                prob = torch.softmax(logits - logits.max(axis=-1, keepdim=True).values, dim=-1)
                pi_action = torch.multinomial(prob, 1)
            else:
                logits = [_logit / math.exp(self._log_alpha.item()) for _logit in output['logit']]
                prob = [
                    torch.softmax(_logits - _logits.max(axis=-1, keepdim=True).values, dim=-1) for _logits in logits
                ]
                pi_action = [torch.multinomial(_prob, 1) for _prob in prob]
        else:
            if isinstance(self._action_dim, int):
                pi_action = torch.randint(0, self._action_dim, (output["logit"].shape[0], ))
            else:
                pi_action = [torch.randint(0, d, (output["logit"][0].shape[0], )) for d in self._action_dim]

        output['action'] = pi_action
        if self._use_cuda:
            output = to_device(output, 'cpu')
        output = default_decollate(output)
        return {i: d for i, d in zip(data_id, output)}

    def _process_transition(self, obs: Any, armor_output: dict, timestep: namedtuple) -> dict:
        r"""
        Overview:
            Generate dict type transition data from inputs.
        Arguments:
            - obs (:obj:`Any`): Env observation
            - armor_output (:obj:`dict`): Output of collect armor, including at least ['action']
            - timestep (:obj:`namedtuple`): Output after env step, including at least ['obs', 'reward', 'done'] \
                (here 'obs' indicates obs after env step, i.e. next_obs).
        Return:
            - transition (:obj:`Dict[str, Any]`): Dict type transition data.
        """
        transition = {
            'obs': obs,
            'next_obs': timestep.obs,
            'action': armor_output['action'],
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
            Init eval armor, which use argmax for selecting action
        """
        self._eval_model = model_wrap(self._model, wrapper_name='argmax_sample')
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
        if self._use_cuda:
            data = to_device(data, self._device)
        self._eval_model.eval()
        with torch.no_grad():
            output = self._eval_model.forward(data)
        if self._use_cuda:
            output = to_device(output, 'cpu')
        output = default_decollate(output)
        return {i: d for i, d in zip(data_id, output)}

    def _create_model(self, cfg: dict, model: Optional[torch.nn.Module] = None) -> torch.nn.Module:
        assert model is None
        return SQNModel(**cfg.model)

    def _monitor_vars_learn(self) -> List[str]:
        r"""
        Overview:
            Return variables' name if variables are to used in monitor.
        Returns:
            - vars (:obj:`List[str]`): Variables' name list.
        """
        return ['alpha_loss', 'alpha', 'entropy', 'q_loss', 'q_value']
