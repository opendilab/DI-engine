from nervex.computation_graph import BaseCompGraph
from nervex.rl_utils import ppo_data, ppo_error
from nervex.worker.agent import BaseAgent


class AtariPpoGraph(BaseCompGraph):

    def __init__(self, cfg: dict) -> None:
        self._lambda = cfg.ppo.gae_lambda
        self._clip_ratio = cfg.ppo.clip_ratio

        self._value_weight = cfg.ppo.value_weight
        self._entropy_weight = cfg.ppo.entropy_weight

    def forward(self, data: dict, agent: BaseAgent) -> dict:
        output = agent.forward(data['obs'])
        weight = data.get('IS', None)
        adv = data['adv']
        # calculate return = adv + value
        return_ = output['value'] + adv
        # norm adv in total train_batch
        mean, std = adv.mean(), adv.std()
        adv = (adv - mean) / (std + 1e-8)
        # calculate ppo error
        data = ppo_data(
            output['logit'], data['logit'], data['action'], output['value'], data['value'], adv, return_, weight
        )
        ppo_loss, ppo_info = ppo_error(data, self._clip_ratio)
        total_loss = ppo_loss.policy_loss + self._value_weight * ppo_loss.value_loss - self._entropy_weight * ppo_loss.entropy_loss

        return {
            'total_loss': total_loss,
            'policy_loss': ppo_loss.policy_loss.item(),
            'value_loss': ppo_loss.value_loss.item(),
            'entropy_loss': ppo_loss.entropy_loss.item(),
            'approx_kl': ppo_info.approx_kl,
            'clipfrac': ppo_info.clipfrac,
        }

    def __repr__(self) -> str:
        return "AtariPpoGraph"

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
