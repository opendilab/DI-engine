from nervex.computation_graph import BaseCompGraph
from nervex.rl_utils import a2c_data, a2c_error
from nervex.worker.agent import BaseAgent


class AtariA2CGraph(BaseCompGraph):

    def __init__(self, cfg: dict) -> None:
        self._value_weight = cfg.a2c.value_weight
        self._entropy_weight = cfg.a2c.entropy_weight

    def forward(self, data: dict, agent: BaseAgent) -> dict:
        output = agent.forward(data['obs'])
        weight = data.get('IS', None)
        adv = data['adv']
        # calculate return = adv + value
        return_ = output['value'] + adv
        # norm adv in total train_batch
        mean, std = adv.mean(), adv.std()
        adv = (adv - mean) / (std + 1e-8)
        # calculate a2c error
        data = a2c_data(output['logit'], data['action'], output['value'], adv, return_, weight)
        a2c_loss, a2c_info = a2c_error(data)
        total_loss = a2c_loss.policy_loss + self._value_weight * a2c_loss.value_loss - self._entropy_weight * a2c_loss.entropy_loss

        return {
            'total_loss': total_loss,
            'policy_loss': a2c_loss.policy_loss.item(),
            'value_loss': a2c_loss.value_loss.item(),
            'entropy_loss': a2c_loss.entropy_loss.item(),
        }

    def __repr__(self) -> str:
        return "AtariA2CGraph"

    def register_stats(self, recorder: 'VariableRecorder', tb_logger: 'TensorBoardLogger') -> None:  # noqa
        recorder.register_var('total_loss')
        recorder.register_var('policy_loss')
        recorder.register_var('value_loss')
        recorder.register_var('entropy_loss')
        recorder.register_var('approx_kl')
        recorder.register_var('clipfrac')
        tb_logger.register_var('total_loss')
        tb_logger.register_var('policy_loss')
        tb_logger.register_var('value_loss')
        tb_logger.register_var('entropy_loss')
        tb_logger.register_var('approx_kl')
        tb_logger.register_var('clipfrac')
