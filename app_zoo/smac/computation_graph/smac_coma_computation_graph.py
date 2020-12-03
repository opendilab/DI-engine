from nervex.computation_graph import BaseCompGraph
from nervex.rl_utils import coma_data, coma_error
from typing import List
from nervex.worker.agent import BaseAgent
import torch
from copy import deepcopy
from nervex.rl_utils import generalized_lambda_returns
from nervex.torch_utils import one_hot


class SMACComaGraph(BaseCompGraph):

    def __init__(self, cfg: dict) -> None:
        self._value_weight = cfg.coma.value_weight
        self._entropy_weight = cfg.coma.entropy_weight
        self._gamma = cfg.coma.gamma
        self._lambda_ = cfg.coma.lambda_

    def forward(self, data: dict, learner_agent: BaseAgent, actor_agents: List[BaseAgent], ) -> dict:
        # note: shape is (T, B,) not (B, )
        assert set(data.keys()) > set(['obs', 'action', 'last_action', 'reward'])
        outputs = [
            actor_agents[i].forward(data['obs']['agent_state'][:, :, i])['logit'] for i in range(len(actor_agents))
        ]
        output_logit = torch.stack(outputs, 2)
        outputs = learner_agent.forward(data)
        q_val = outputs['total_q']
        weight = data.get('IS', None)
        baseline = (output_logit * q_val).sum(-1)
        action = data['action']
        action_dim = output_logit.shape[3]
        action_onehot = one_hot(action, action_dim)
        q_taken = torch.sum(q_val * action_onehot, -1)
        adv = (q_taken - baseline).detach()
        # calculate return_
        target_q_val = learner_agent.target_forward(data)
        target_q_taken = torch.sum(target_q_val * action_onehot, -1)

        return_ = generalized_lambda_returns(target_q_taken, data['reward'][:-1], self._gamma, self._lambda_)
        # calculate coma error
        data = coma_data(output_logit, data['action'], q_val, adv, return_, weight, data['mask'])
        coma_loss = coma_error(data)
        total_loss = coma_loss.policy_loss + self._value_weight * coma_loss.q_val_loss - self._entropy_weight * \
            coma_loss.entropy_loss

        return {
            'total_loss': total_loss,
            'policy_loss': coma_loss.policy_loss.item(),
            'value_loss': coma_loss.q_val_loss.item(),
            'entropy_loss': coma_loss.entropy_loss.item(),
            'adv_abs_max': adv.abs().max().item(),
        }

    def __repr__(self) -> str:
        return "SMACComaGraph"

    def register_stats(self, recorder: 'VariableRecorder', tb_logger: 'TensorBoardLogger') -> None:  # noqa
        recorder.register_var('total_loss')
        recorder.register_var('policy_loss')
        recorder.register_var('value_loss')
        recorder.register_var('entropy_loss')
        recorder.register_var('adv_abs_max')
        tb_logger.register_var('total_loss')
        tb_logger.register_var('policy_loss')
        tb_logger.register_var('value_loss')
        tb_logger.register_var('entropy_loss')
        tb_logger.register_var('adv_abs_max')
