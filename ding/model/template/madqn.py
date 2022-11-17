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
            global_cooperation: bool = True,
            lstm_type: str = 'gru',
            dueling: bool = False
    ) -> None:
        super(MADQN, self).__init__()
        self.current = QMix(
            agent_num=agent_num,
            obs_shape=obs_shape,
            action_shape=action_shape,
            hidden_size_list=hidden_size_list,
            global_obs_shape=global_obs_shape,
            mixer=mixer,
            lstm_type=lstm_type,
            dueling=dueling
        )
        self.global_cooperation = global_cooperation
        if self.global_cooperation:
            cooperation_obs_shape = global_obs_shape
        else:
            cooperation_obs_shape = obs_shape
        self.cooperation = QMix(
            agent_num=agent_num,
            obs_shape=cooperation_obs_shape,
            action_shape=action_shape,
            hidden_size_list=hidden_size_list,
            global_obs_shape=global_obs_shape,
            mixer=mixer,
            lstm_type=lstm_type,
            dueling=dueling
        )

    def forward(self, data: dict, cooperation: bool = False, single_step: bool = True) -> dict:
        if cooperation:
            if self.global_cooperation:
                data['obs']['agent_state'] = data['obs']['global_state']
            return self.cooperation(data, single_step=single_step)
        else:
            return self.current(data, single_step=single_step)
