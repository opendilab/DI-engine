from nervex.computation_graph import BaseCompGraph
from nervex.rl_utils import q_1step_td_data_continuous, q_1step_td_error_continuous
from nervex.worker.agent import BaseAgent


class PendulumDdpgGraph(BaseCompGraph):

    def __init__(self, cfg: dict) -> None:
        self._gamma = cfg.ddpg.discount_factor
        self._actor_update_freq = cfg.actor_update_freq
        self._forward_cnt = 0

    def forward(self, data: dict, agent: BaseAgent) -> dict:
        obs = data.get('obs')
        next_obs = data.get('next_obs')
        reward = data.get('reward')
        action = data.get('action')
        done = data.get('done').float()
        weights = data.get('weights', None)

        # q
        q_value = agent.forward(data, mode='compute_q')['q']
        # target_q
        next_data = {'obs': next_obs}
        next_action = agent.forward(next_data, mode='compute_action')
        next_data['action'] = next_action
        if agent.is_double:
            target_q_value = agent.target_forward(next_data)['q']
        else:
            target_q_value = agent.forward(next_data)['q']
        # critic_loss: q 1step td error
        data = q_1step_td_data_continuous(q_value, target_q_value, action, reward, done)
        critic_loss = q_1step_td_error_continuous(data, self._gamma, weights)
        actor_update = self._forward_cnt % self._actor_update_freq == 0
        # actor_loss: q grad ascent
        if actor_update:
            actor_loss = -agent.forward(data, mode='optimize_actor')['q'].mean()

        if agent.is_double:
            agent.update_target_network(agent.state_dict()['model'])
        self._forward_cnt += 1
        loss_dict = {}
        if actor_update:
            loss_dict['actor_loss'] = actor_loss
        for i, loss in enumerate(critic_loss):
            loss_dict['critic{}_loss'.format(str(i + 1))] = loss
        return loss_dict

    def __repr__(self) -> str:
        return "PendulumDdpgGraph"

    def register_stats(self, recorder: 'VariableRecorder', tb_logger: 'TensorBoardLogger') -> None:  # noqa
        recorder.register_var('actor_loss')
        recorder.register_var('critic1_loss')
        recorder.register_var('critic2_loss')
        tb_logger.register_var('actor_loss')
        tb_logger.register_var('critic1_loss')
        tb_logger.register_var('critic2_loss')
