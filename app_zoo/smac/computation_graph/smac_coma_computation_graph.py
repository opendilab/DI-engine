from typing import List
import torch
from copy import deepcopy
from nervex.worker.agent import BaseAgent
from nervex.computation_graph import BaseCompGraph
from nervex.rl_utils import coma_data, coma_error
from nervex.torch_utils import one_hot


class SMACComaGraph(BaseCompGraph):

    def __init__(self, cfg: dict) -> None:
        self._value_weight = cfg.coma.value_weight
        self._entropy_weight = cfg.coma.entropy_weight
        self._gamma = cfg.coma.discount_factor
        self._lambda = cfg.coma.td_lambda

    def forward(self, data: dict, agent: BaseAgent) -> dict:
        # note: shape is (T, B,) not (B, )
        assert set(data.keys()) > set(['obs', 'action', 'reward'])
        weight = data.get('weight', None)
        agent.reset(state=data['prev_state'][0])
        # prepare data
        q_value = agent.forward(data, param={'mode': 'compute_q_value'})['q_value']
        target_q_value = agent.target_forward(data, param={'mode': 'compute_q_value'})['q_value']
        logit = agent.forward(data, param={'mode': 'compute_action'})['logit']

        # calculate coma error
        data = coma_data(logit, data['action'], q_value, target_q_value, data['reward'], weight)
        coma_loss = coma_error(data, self._gamma, self._lambda)
        total_loss = coma_loss.policy_loss + self._value_weight * coma_loss.q_value_loss - self._entropy_weight * \
            coma_loss.entropy_loss

        return {
            'total_loss': total_loss,
            'policy_loss': coma_loss.policy_loss.item(),
            'value_loss': coma_loss.q_value_loss.item(),
            'entropy_loss': coma_loss.entropy_loss.item(),
        }

    def __repr__(self) -> str:
        return "SMACComaGraph"

    def register_stats(self, recorder: 'VariableRecorder', tb_logger: 'TensorBoardLogger') -> None:  # noqa
        recorder.register_var('total_loss')
        recorder.register_var('policy_loss')
        recorder.register_var('value_loss')
        recorder.register_var('entropy_loss')
        tb_logger.register_var('total_loss')
        tb_logger.register_var('policy_loss')
        tb_logger.register_var('value_loss')
        tb_logger.register_var('entropy_loss')
