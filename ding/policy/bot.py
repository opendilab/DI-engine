from collections import namedtuple
from typing import Optional, List, Dict, Any, Tuple, Union
import torch
import copy
import numpy as np
from ding.torch_utils import to_device
from ding.rl_utils import get_train_sample
from ding.utils import POLICY_REGISTRY
from ding.utils.data import default_collate, default_decollate
from .base_policy import Policy


@POLICY_REGISTRY.register('bot')
class Botpolicy(Policy):
    r"""
    Overview:
        Policy class of on policy version PPO algorithm.
    """
    config = dict(
        # (str) RL policy register name (refer to function "POLICY_REGISTRY").
        type='bot',
        # (bool) Whether to use cuda for network.
        cuda=False,
        learn=dict(
            # (bool) Whether to use multi gpu
            multi_gpu=False,
        ),
        collect=dict(
            # (int) Only one of [n_sample, n_episode] shoule be set
            # n_sample=64,
            # (int) Cut trajectories into pieces with length "unroll_len".
            unroll_len=1,
        ),
        eval=dict(),
    )
    
    def __init__(
            self,
            cfg: dict,
            model: Optional[Union[type, torch.nn.Module]] = None,
            enable_field: Optional[List[str]] = None
    ) -> None:
        self._cfg = cfg
        self._on_policy = self._cfg.on_policy
        if enable_field is None:
            self._enable_field = self.total_field
        else:
            self._enable_field = enable_field
        assert set(self._enable_field).issubset(self.total_field), self._enable_field

        if len(set(self._enable_field).intersection(set(['learn', 'collect', 'eval']))) > 0:
            self._cuda = cfg.cuda and torch.cuda.is_available()
            # now only support multi-gpu for only enable learn mode
            if len(set(self._enable_field).intersection(set(['learn']))) > 0:
                self._rank = get_rank() if self._cfg.learn.multi_gpu else 0
                if self._cuda:
                    torch.cuda.set_device(self._rank % torch.cuda.device_count())
                    model.cuda()
                if self._cfg.learn.multi_gpu:
                    bp_update_sync = self._cfg.learn.get('bp_update_sync', True)
                    self._bp_update_sync = bp_update_sync
                    self._init_multi_gpu_setting(model, bp_update_sync)
            else:
                self._rank = 0
                if self._cuda:
                    torch.cuda.set_device(self._rank % torch.cuda.device_count())
                    model.cuda()
            self._model = model
            self._device = 'cuda:{}'.format(self._rank % torch.cuda.device_count()) if self._cuda else 'cpu'
        else:
            self._cuda = False
            self._rank = 0
            self._device = 'cpu'

        for field in self._enable_field:
            getattr(self, '_init_' + field)()

    def _init_learn(self) -> None:
        pass

    def _forward_learn(self, data: Dict[str, Any]) -> Dict[str, Any]:
        pass

    def _state_dict_learn(self) -> Dict[str, Any]:
        pass

    def _load_state_dict_learn(self, state_dict: Dict[str, Any]) -> None:
        pass
    
    def _init_collect(self) -> None:
        """
        Overview:
            Collect mode init method. Called by ``self.__init__``, initialize algorithm arguments and collect_model, \
            enable the eps_greedy_sample for exploration.
        """
        self._unroll_len = self._cfg.collect.unroll_len
        self._collect_model = self._model
        #self._collect_model.reset()

    def _forward_collect(self, data: dict) -> dict:
        """
        Overview:
            Forward function of collect mode.
        Arguments:
            - data (:obj:`Dict[str, Any]`): Dict type data, stacked env data for predicting policy_output(action), \
                values are torch.Tensor or np.ndarray or dict/list combinations, keys are env_id indicated by integer.
        Returns:
            - output (:obj:`Dict[int, Any]`): Dict type data, including at least inferred action according to input obs.
        ReturnsKeys
            - necessary: ``action``
        """
        data_id = list(data.keys())
        data = default_collate(list(data.values()))
        if self._cuda:
            data = to_device(data, self._device)
        output = self._collect_model.predict(data)
        # a dict{env_id: {'action': np.array(0-5)}}
        return output

    def _process_transition(self, obs: Any, model_output: dict, timestep: namedtuple) -> dict:
        """
        Overview:
               Generate dict type transition data from inputs.
        Arguments:
                - obs (:obj:`Any`): Env observation
                - model_output (:obj:`dict`): Output of collect model, including at least ['action']
                - timestep (:obj:`namedtuple`): Output after env step, including at least ['obs', 'reward', 'done']\
                       (here 'obs' indicates obs after env step).
        Returns:
               - transition (:obj:`dict`): Dict type transition data.
        """
        transition = {
            'obs': obs,
            'next_obs': timestep.obs,
            'action': model_output['action'],
            'reward': timestep.reward,
            'done': timestep.done,
        }
        return transition

    def _get_train_sample(self, data: list) -> Union[None, List[Any]]:
        return get_train_sample(data, self._unroll_len)

    def _init_eval(self) -> None:
        pass
        
    def _forward_eval(self, data: dict) -> dict:
        pass
        
    def default_model(self) -> Tuple[str, List[str]]:
        return 'bot', ['ding.model.template.bot']

    def _monitor_vars_learn(self) -> List[str]:
        pass