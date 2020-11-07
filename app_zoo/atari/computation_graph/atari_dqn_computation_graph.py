from nervex.computation_graph import BaseCompGraph
from nervex.rl_utils import td_data, one_step_td_error
from nervex.worker.agent import BaseAgent


class AtariDqnGraph(BaseCompGraph):

    def __init__(self, cfg: dict) -> None:
        self._gamma = cfg.dqn.discount_factor

    def forward(self, data: dict, agent: BaseAgent) -> dict:
        obs = data.get('obs')
        next_obs = data.get('next_obs')
        reward = data.get('reward')
        action = data.get('action')
        done = data.get('done').float()
        weights = data.get('weights', None)

        q_value = agent.forward(obs)
        if agent.is_double:
            target_q_value = agent.target_forward(next_obs)
        else:
            target_q_value = agent.forward(next_obs)

        data = td_data(q_value, target_q_value, action, reward, done)
        loss = one_step_td_error(data, self._gamma, weights)
        if agent.is_double:
            agent.update_target_network(agent.state_dict()['model'])
        return {'total_loss': loss}

    def __repr__(self) -> str:
        return "AtariDqnGraph"

    def register_stats(self, recorder: 'VariableRecorder', tb_logger: 'TensorBoardLogger') -> None:  # noqa
        recorder.register_var('total_loss')
        tb_logger.register_var('total_loss')
