from typing import Dict, Any
import torch
from ding.rl_utils import q_nstep_td_data, q_nstep_td_error
from ding.policy import DQNPolicy
from ding.utils import POLICY_REGISTRY
from ding.policy.common_utils import default_preprocess_learn
from ding.torch_utils import to_device


@POLICY_REGISTRY.register('md_dqn')
class MultiDiscreteDQNPolicy(DQNPolicy):
    r"""
    Overview:
        Policy class of Multi-discrete action space DQN algorithm.
    """

    def _forward_learn(self, data: dict) -> Dict[str, Any]:
        """
        Overview:
            Forward computation of learn mode(updating policy). It supports both single and multi-discrete action \
                space. It depends on whether the ``q_value`` is a list.
        Arguments:
            - data (:obj:`Dict[str, Any]`): Dict type data, a batch of data for training, values are torch.Tensor or \
                np.ndarray or dict/list combinations.
        Returns:
            - info_dict (:obj:`Dict[str, Any]`): Dict type data, a info dict indicated training result, which will be \
                recorded in text log and tensorboard, values are python scalar or a list of scalars.
        ArgumentsKeys:
            - necessary: ``obs``, ``action``, ``reward``, ``next_obs``, ``done``
            - optional: ``value_gamma``, ``IS``
        ReturnsKeys:
            - necessary: ``cur_lr``, ``total_loss``, ``priority``
            - optional: ``action_distribution``
        """
        data = default_preprocess_learn(
            data,
            use_priority=self._priority,
            use_priority_IS_weight=self._cfg.priority_IS_weight,
            ignore_done=self._cfg.learn.ignore_done,
            use_nstep=True
        )
        if self._cuda:
            data = to_device(data, self._device)
        # ====================
        # Q-learning forward
        # ====================
        self._learn_model.train()
        self._target_model.train()
        # Current q value (main model)
        q_value = self._learn_model.forward(data['obs'])['logit']
        # Target q value
        with torch.no_grad():
            target_q_value = self._target_model.forward(data['next_obs'])['logit']
            # Max q value action (main model)
            target_q_action = self._learn_model.forward(data['next_obs'])['action']

        value_gamma = data.get('value_gamma')
        if isinstance(q_value, list):
            act_num = len(q_value)
            loss, td_error_per_sample = [], []
            q_value_list = []
            for i in range(act_num):
                td_data = q_nstep_td_data(
                    q_value[i], target_q_value[i], data['action'][i], target_q_action[i], data['reward'], data['done'],
                    data['weight']
                )
                loss_, td_error_per_sample_ = q_nstep_td_error(
                    td_data, self._gamma, nstep=self._nstep, value_gamma=value_gamma
                )
                loss.append(loss_)
                td_error_per_sample.append(td_error_per_sample_.abs())
                q_value_list.append(q_value[i].mean().item())
            loss = sum(loss) / (len(loss) + 1e-8)
            td_error_per_sample = sum(td_error_per_sample) / (len(td_error_per_sample) + 1e-8)
            q_value_mean = sum(q_value_list) / act_num
        else:
            data_n = q_nstep_td_data(
                q_value, target_q_value, data['action'], target_q_action, data['reward'], data['done'], data['weight']
            )
            loss, td_error_per_sample = q_nstep_td_error(
                data_n, self._gamma, nstep=self._nstep, value_gamma=value_gamma
            )
            q_value_mean = q_value.mean().item()

        # ====================
        # Q-learning update
        # ====================
        self._optimizer.zero_grad()
        loss.backward()
        if self._cfg.learn.multi_gpu:
            self.sync_gradients(self._learn_model)
        self._optimizer.step()

        # =============
        # after update
        # =============
        self._target_model.update(self._learn_model.state_dict())
        return {
            'cur_lr': self._optimizer.defaults['lr'],
            'total_loss': loss.item(),
            'q_value_mean': q_value_mean,
            'priority': td_error_per_sample.abs().tolist(),
        }
