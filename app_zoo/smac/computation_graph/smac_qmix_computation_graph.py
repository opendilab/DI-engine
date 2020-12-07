from nervex.computation_graph import BaseCompGraph
from nervex.rl_utils import v_1step_td_error, v_1step_td_data
from nervex.worker.agent import BaseAgent


class SMACQMixGraph(BaseCompGraph):

    def __init__(self, cfg: dict) -> None:
        self._gamma = cfg.qmix.discount_factor

    def forward(self, data: dict, agent: BaseAgent) -> dict:
        reward, action = data['reward'], data['action']
        done = data['done'].float()
        weights = data.get('weights', None)
        agent.reset(state=data['prev_state'][0])

        inputs = {'obs': data['obs'], 'action': action}
        total_q = agent.forward(inputs, param={'single_step': False})['total_q']

        next_inputs = {'obs': data['next_obs']}
        target_total_q = agent.target_forward(next_inputs, param={'single_step': False})['total_q']

        data = v_1step_td_data(total_q, target_total_q, reward, done, weights)
        loss = v_1step_td_error(data, self._gamma)
        agent.target_update(agent.state_dict()['model'])
        return {'total_loss': loss}

    def __repr__(self) -> str:
        return "SMACQMixGraph"

    def register_stats(self, recorder: 'VariableRecorder', tb_logger: 'TensorBoardLogger') -> None:  # noqa
        recorder.register_var('total_loss')
        tb_logger.register_var('total_loss')
        # total_q, target_total_q
