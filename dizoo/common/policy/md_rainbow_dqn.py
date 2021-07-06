from typing import Dict, Any
import torch
from ding.rl_utils import dist_nstep_td_data, dist_nstep_td_error, dist_1step_td_data, dist_1step_td_error
from ding.policy import RainbowDQNPolicy
from ding.utils import POLICY_REGISTRY


@POLICY_REGISTRY.register('md_rainbow_dqn')
class MultiDiscreteRainbowDQNPolicy(RainbowDQNPolicy):

    def _forward_learn(self, data: dict) -> Dict[str, Any]:
        """
        Overview:
            Forward and backward function of learn mode, acquire the data and calculate the loss and\
            optimize learner model

        Arguments:
            - data (:obj:`dict`): Dict type data, including at least ['obs', 'next_obs', 'reward', 'action']

        Returns:
            - info_dict (:obj:`Dict[str, Any]`): Including cur_lr and total_loss
                - cur_lr (:obj:`float`): current learning rate
                - total_loss (:obj:`float`): the calculated loss
        """
        # ====================
        # Rainbow forward
        # ====================
        reward = data['reward']
        if len(reward.shape) == 1:
            reward = reward.unsqueeze(1)
        assert reward.shape == (self._cfg.learn.batch_size, self._nstep), reward.shape
        reward = reward.permute(1, 0).contiguous()
        # reset noise of noisenet for both main armor and target armor
        self._reset_noise(self._armor.model)
        self._reset_noise(self._armor.target_model)
        q_dist = self._armor.forward(data['obs'])['distribution']
        with torch.no_grad():
            target_q_dist = self._armor.target_forward(data['next_obs'])['distribution']
            self._reset_noise(self._armor.model)
            target_q_action = self._armor.forward(data['next_obs'])['action']
        if isinstance(q_dist, torch.Tensor):
            td_data = dist_nstep_td_data(
                q_dist, target_q_dist, data['action'], target_q_action, reward, data['done'], data['weight']
            )
            loss, td_error_per_sample = dist_nstep_td_error(
                td_data, self._gamma, self._v_min, self._v_max, self._n_atom, nstep=self._nstep
            )
        else:
            tl_num = len(q_dist)
            losses = []
            td_error_per_samples = []
            for i in range(tl_num):
                td_data = dist_nstep_td_data(
                    q_dist[i], target_q_dist[i], data['action'][i], target_q_action[i], reward, data['done'],
                    data['weight']
                )
                td_loss, td_error_per_sample = dist_nstep_td_error(
                    td_data, self._gamma, self._v_min, self._v_max, self._n_atom, nstep=self._nstep
                )
                losses.append(td_loss)
                td_error_per_samples.append(td_error_per_sample)
            loss = sum(losses) / (len(losses) + 1e-8)
            td_error_per_sample_mean = sum(td_error_per_samples)
        # ====================
        # Rainbow update
        # ====================
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        # =============
        # after update
        # =============
        self._armor.target_update(self._armor.state_dict()['model'])
        return {
            'cur_lr': self._optimizer.defaults['lr'],
            'total_loss': loss.item(),
            'priority': td_error_per_sample_mean.abs().tolist(),
        }
