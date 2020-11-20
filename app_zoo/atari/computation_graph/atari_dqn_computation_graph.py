from nervex.computation_graph import BaseCompGraph
from nervex.rl_utils import q_1step_td_data, q_1step_td_error
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

        q_value = agent.forward(obs)['logit']
        if agent.is_double:
            target_q_value = agent.target_forward(next_obs)['logit']
        else:
            target_q_value = agent.forward(next_obs)['logit']

        data = q_1step_td_data(q_value, target_q_value, action, reward, done)
        loss = q_1step_td_error(data, self._gamma, weights)
        if agent.is_double:
            agent.target_update(agent.state_dict()['model'])
        return {'total_loss': loss}

    def __repr__(self) -> str:
        return "AtariDqnGraph"
