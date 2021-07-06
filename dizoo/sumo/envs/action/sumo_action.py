import torch

from ding.envs.common import EnvElement
from ding.torch_utils import tensor_to_list


class SumoRawAction(EnvElement):
    r"""
    Overview:
        the action element of Sumo enviroment

    Interface:
        _init, _from_agent_processor
    """
    _name = "SumoRawAction"

    def _init(self, cfg):
        r"""
        Overview:
            init the sumo action environment with the given config file
        Arguments:
            - cfg(:obj:`EasyDict`): config, you can refer to `envs/sumo/sumo_env_default_config.yaml`
        """
        self._cfg = cfg
        self._tls_green_action = cfg.tls_green_action
        self._tls_yellow_action = cfg.tls_yellow_action
        self._tls = list(self._tls_green_action.keys())
        self._shape = {k: len(v) for k, v in self._tls_green_action.items()}
        self._value = {
            k: {
                'min': 0,
                'max': self._shape[k] - 1,
                'dtype': int,
                'dinfo': 'int'
            }
            for k in self._tls_green_action.keys()
        }
        self._to_agent_processor = None

    def _from_agent_processor(self, data):
        r"""
        Overview:
            get the raw_action and return corresponding action
        Arguments:
            - data (:obj:`dict`): for data info you can reference the example below

            Example:
                >>> data = {'htxdj_wjj': {'action': torch.tensor([0]), 'last_action': torch.tensor([0])},
                ...         'haxl_wjj': {'action': torch.tensor([0]), 'last_action': torch.tensor([1])},
                ...         'haxl_htxdj': {'action': torch.tensor([0]), 'last_action': torch.tensor([1])}}

        Returns:
            - raw_action(:obj:`dict`): the returned raw_action
        """
        data = tensor_to_list(data)
        raw_action = {k: {} for k in data.keys()}
        for k, v in data.items():
            action, last_action = v['action'], v['last_action']
            if last_action is None:
                yellow_phase = None
            else:
                yellow_phase = self._tls_yellow_action[k][last_action] if action != last_action else None
            raw_action[k]['yellow_phase'] = yellow_phase
            raw_action[k]['green_phase'] = self._tls_green_action[k][action]
        return raw_action

    # override
    def _details(self):
        return 'action dim: {}'.format(self._shape)
