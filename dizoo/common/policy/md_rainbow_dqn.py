from typing import Dict, Any
import torch
from ding.torch_utils import to_device
from ding.rl_utils import dist_nstep_td_data, dist_nstep_td_error, dist_1step_td_data, dist_1step_td_error
from ding.policy import RainbowDQNPolicy
from ding.utils import POLICY_REGISTRY
from ding.policy.common_utils import default_preprocess_learn


@POLICY_REGISTRY.register('md_rainbow_dqn')
class MultiDiscreteRainbowDQNPolicy(RainbowDQNPolicy):
    r"""
    Overview:
        Multi-discrete action space Rainbow DQN algorithms.
    """

    def _forward_learn(self, data: dict) -> Dict[str, Any]:
        """
        Overview:
            Forward and backward function of learn mode, acquire the data and calculate the loss and \
            optimize learner model

        Arguments:
            - data (:obj:`dict`): Dict type data, including at least ['obs', 'next_obs', 'reward', 'action']

        Returns:
            - info_dict (:obj:`Dict[str, Any]`): Including cur_lr, total_loss and priority
                - cur_lr (:obj:`float`): current learning rate
                - total_loss (:obj:`float`): the calculated loss
                - priority (:obj:`list`): the priority of samples
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
        # Rainbow forward
        # ====================
        self._learn_model.train()
        self._target_model.train()
        # reset noise of noisenet for both main model and target model
        self._reset_noise(self._learn_model)
        self._reset_noise(self._target_model)
        q_dist = self._learn_model.forward(data['obs'])['distribution']
        with torch.no_grad():
            target_q_dist = self._target_model.forward(data['next_obs'])['distribution']
            self._reset_noise(self._learn_model)
            target_q_action = self._learn_model.forward(data['next_obs'])['action']

        value_gamma = data.get('value_gamma', None)
        if isinstance(q_dist, torch.Tensor):
            td_data = dist_nstep_td_data(
                q_dist, target_q_dist, data['action'], target_q_action, data['reward'], data['done'], data['weight']
            )
            loss, td_error_per_sample = dist_nstep_td_error(
                td_data,
                self._gamma,
                self._v_min,
                self._v_max,
                self._n_atom,
                nstep=self._nstep,
                value_gamma=value_gamma
            )
        else:
            act_num = len(q_dist)
            losses = []
            td_error_per_samples = []
            for i in range(act_num):
                td_data = dist_nstep_td_data(
                    q_dist[i], target_q_dist[i], data['action'][i], target_q_action[i], data['reward'], data['done'],
                    data['weight']
                )
                td_loss, td_error_per_sample = dist_nstep_td_error(
                    td_data,
                    self._gamma,
                    self._v_min,
                    self._v_max,
                    self._n_atom,
                    nstep=self._nstep,
                    value_gamma=value_gamma
                )
                losses.append(td_loss)
                td_error_per_samples.append(td_error_per_sample)
            loss = sum(losses) / (len(losses) + 1e-8)
            td_error_per_sample_mean = sum(td_error_per_samples) / (len(td_error_per_samples) + 1e-8)
        # ====================
        # Rainbow update
        # ====================
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        # =============
        # after update
        # =============
        self._target_model.update(self._learn_model.state_dict())
        return {
            'cur_lr': self._optimizer.defaults['lr'],
            'total_loss': loss.item(),
            'priority': td_error_per_sample_mean.abs().tolist(),
        }
