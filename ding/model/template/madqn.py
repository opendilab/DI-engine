import torch.nn as nn
from ding.utils import MODEL_REGISTRY
from .qmix import QMix


@MODEL_REGISTRY.register('madqn')
class MADQN(nn.Module):
    def __init__(
            self,
            agent_num: int,
            obs_shape: int,
            action_shape: int,
            hidden_size_list: list,
            global_obs_shape: int = None,
            mixer: bool = False,
            global_boost: bool = True,
            lstm_type: str = 'gru',
            dueling: bool = False
    ) -> None:
        super(SIQL, self).__init__()
        self.current = QMix(agent_num, obs_shape, action_shape, hidden_size_list, global_obs_shape=global_obs_shape, mixer=mixer, lstm_type=lstm_type, dueling=dueling)
        self.global_boost = global_boost
        if self.global_boost:
            boost_obs_shape = global_obs_shape
        else:
            boost_obs_shape = obs_shape
        self.boost = QMix(agent_num, boost_obs_shape, action_shape, hidden_size_list, global_obs_shape=global_obs_shape, mixer=mixer, lstm_type=lstm_type, dueling=dueling)

    def forward(self, data: dict, boost: bool = False, single_step: bool = True) -> dict:
        if boost:
            if self.global_boost:
                data['obs']['agent_state'] = data['obs']['global_state']
            return self.boost(data, single_step=single_step)
        else:
            return self.current(data, single_step=single_step)
