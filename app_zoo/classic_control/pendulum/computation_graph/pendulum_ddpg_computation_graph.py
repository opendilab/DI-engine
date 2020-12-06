import torch
from nervex.computation_graph import BaseCompGraph
from nervex.rl_utils import v_1step_td_data, v_1step_td_error
from nervex.worker.agent import BaseAgent


class PendulumDdpgGraph(BaseCompGraph):

    def __init__(self, cfg: dict) -> None:
        self._gamma = cfg.ddpg.discount_factor
        self._actor_update_freq = cfg.ddpg.get('actor_update_freq', 2)
        self._use_twin_critic = cfg.ddpg.use_twin_critic
        self._forward_cnt = 0
        self._cfg = cfg

    def forward(self, data: dict, agent: BaseAgent) -> dict:
        loss_dict = {}
        if (self._forward_cnt + 1) % self._actor_update_freq == 0:
            actor_loss = -agent.forward(data, param={'mode': 'optimize_actor'})['q_value'].mean()
            loss_dict['actor_loss'] = actor_loss
        else:
            next_obs = data.get('next_obs')
            reward = data.get('reward')
            reward = (reward - reward.mean()) / (reward.std() + 1e-8)
            action = data.get('action')

            q_value = agent.forward(data, param={'mode': 'compute_q'})['q_value']
            next_data = {'obs': next_obs}
            next_action = agent.target_forward(next_data, param={'mode': 'compute_action'})['action']
            next_data['action'] = next_action
            target_q_value = agent.target_forward(next_data, param={'mode': 'compute_q'})['q_value']
            if self._use_twin_critic:
                target_q_value = torch.min(target_q_value[0], target_q_value[1])
                data = v_1step_td_data(q_value[0], target_q_value, reward, None, None)
                critic_loss = v_1step_td_error(data, self._gamma)
                loss_dict['critic_loss'] = critic_loss
                data_twin = v_1step_td_data(q_value[1], target_q_value, reward, None, None)
                critic_twin_loss = v_1step_td_error(data_twin, self._gamma)
                loss_dict['critic_twin_loss'] = critic_twin_loss
            else:
                data = v_1step_td_data(q_value, target_q_value, reward, None, None)
                critic_loss = v_1step_td_error(data, self._gamma)
                loss_dict['critic_loss'] = critic_loss
        agent.target_update(agent.state_dict()['model'])
        loss_dict['total_loss'] = sum(loss_dict.values())
        self._forward_cnt += 1
        return loss_dict

    def __repr__(self) -> str:
        return "PendulumDdpgGraph"

    def register_stats(self, recorder: 'VariableRecorder', tb_logger: 'TensorBoardLogger') -> None:  # noqa
        recorder.register_var('actor_loss')
        recorder.register_var('critic_loss')
        recorder.register_var('total_loss')
        tb_logger.register_var('actor_loss')
        tb_logger.register_var('critic_loss')
        tb_logger.register_var('total_loss')
        if self._use_twin_critic:
            recorder.register_var('critic_twin_loss')
            tb_logger.register_var('critic_twin_loss')
