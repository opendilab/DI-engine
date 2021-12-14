from typing import Any, Tuple, Callable, Optional, List, Dict
from abc import ABC

import numpy as np
import torch
from ding.torch_utils import get_tensor_data
from ding.rl_utils import create_noise_generator
from torch.distributions import Categorical
from ding.utils.data import default_collate


class IModelWrapper(ABC):
    r"""
    Overview:
        the base class of Model Wrappers
    Interfaces:
        register
    """

    def __init__(self, model: Any) -> None:
        self._model = model

    def __getattr__(self, key: str) -> Any:
        r"""
        Overview:
            Get the attrbute in model.
        Arguments:
            - key (:obj:`str`): The key to query.
        Returns:
            - ret (:obj:`Any`): The queried attribute.
        """
        return getattr(self._model, key)

    def info(self, attr_name):
        r"""
        Overview:
            get info of attr_name
        """
        if attr_name in dir(self):
            if isinstance(self._model, IModelWrapper):
                return '{} {}'.format(self.__class__.__name__, self._model.info(attr_name))
            else:
                if attr_name in dir(self._model):
                    return '{} {}'.format(self.__class__.__name__, self._model.__class__.__name__)
                else:
                    return '{}'.format(self.__class__.__name__)
        else:
            if isinstance(self._model, IModelWrapper):
                return '{}'.format(self._model.info(attr_name))
            else:
                return '{}'.format(self._model.__class__.__name__)


class BaseModelWrapper(IModelWrapper):
    r"""
    Overview:
        the base class of Model Wrappers
    Interfaces:
        register
    """

    def reset(self, data_id: List[int] = None) -> None:
        r"""
        Overview
            the reset function that the Model Wrappers with states should implement
            used to reset the stored states
        """
        pass


class HiddenStateWrapper(IModelWrapper):

    def __init__(
            self, model: Any, state_num: int, save_prev_state: bool = False, init_fn: Callable = lambda: None,
    ) -> None:
        """
        Overview:
            Maintain the hidden state for RNN-base model. Each sample in a batch has its own state. \
            Init the maintain state and state function; Then wrap the ``model.forward`` method with auto \
            saved data ['prev_state'] input, and create the ``model.reset`` method.
        Arguments:
            - model(:obj:`Any`): Wrapped model class, should contain forward method.
            - state_num (:obj:`int`): Number of states to process.
            - save_prev_state (:obj:`bool`): Whether to output the prev state in output['prev_state'].
            - init_fn (:obj:`Callable`): The function which is used to init every hidden state when init and reset. \
                Default return None for hidden states.
        .. note::
            1. This helper must deal with an actual batch with some parts of samples, e.g: 6 samples of state_num 8.
            2. This helper must deal with the single sample state reset.
        """
        super().__init__(model)
        self._state_num = state_num
        self._state = {i: init_fn() for i in range(state_num)}
        self._save_prev_state = save_prev_state
        self._init_fn = init_fn

    def forward(self, data, **kwargs):
        state_id = kwargs.pop('data_id', None)
        valid_id = kwargs.pop('valid_id', None)
        data, state_info = self.before_forward(data, state_id)
        output = self._model.forward(data, **kwargs)
        h = output.pop('next_state', None)
        if h is not None:
            self.after_forward(h, state_info, valid_id)
        if self._save_prev_state:
            prev_state = get_tensor_data(data['prev_state'])
            output['prev_state'] = prev_state
        return output

    def reset(self, *args, **kwargs):
        state = kwargs.pop('state', None)
        state_id = kwargs.get('data_id', None)
        self.reset_state(state, state_id)
        if hasattr(self._model, 'reset'):
            return self._model.reset(*args, **kwargs)

    def reset_state(self, state: Optional[list] = None, state_id: Optional[list] = None) -> None:
        if state_id is None:
            state_id = [i for i in range(self._state_num)]
        if state is None:
            state = [self._init_fn() for i in range(len(state_id))]
        assert len(state) == len(state_id), '{}/{}'.format(len(state), len(state_id))
        for idx, s in zip(state_id, state):
            self._state[idx] = s

    def before_forward(self, data: dict, state_id: Optional[list]) -> Tuple[dict, dict]:
        if state_id is None:
            state_id = [i for i in range(self._state_num)]

        state_info = {idx: self._state[idx] for idx in state_id}
        data['prev_state'] = list(state_info.values())
        return data, state_info

    def after_forward(self, h: Any, state_info: dict, valid_id: Optional[list] = None) -> None:
        assert len(h) == len(state_info), '{}/{}'.format(len(h), len(state_info))
        for i, idx in enumerate(state_info.keys()):
            if valid_id is None:
                self._state[idx] = h[i]
            else:
                if idx in valid_id:
                    self._state[idx] = h[i]


class TransformerWrapper(IModelWrapper):

    def __init__(
            self, model: Any, seq_len: int, init_fn: Callable = lambda: None
    ) -> None:
        """
        Overview:
            Given N the length of the sequences received by a Transformer model, maintain the last N-1 input
            observations. In this way we can provide at each step all the observations needed by Transformer to
            compute its output. We need this because some methods such as 'collect' and 'evaluate' only provide the
            model 1 observation per step and don't have memory of past observations, but Transformer needs a sequence
            of N observations. The wrapped ``model.forward`` will save the input observation in a FIFO memory of length
            N and the wrapped ``model.reset`` will reset the memory. The empty memory spaces will be initialized with
            'init_fn' or the input observation by calling the method ``model.reset_memory``.
        Arguments:
            - model (:obj:`Any`): Wrapped model class, should contain forward method.
            - seq_len (:obj:`int`): Number of past observations to remember.
            - init_fn (:obj:`Callable`): The function which is used to init every memory locations when init and reset.
        """
        super().__init__(model)
        self.seq_len = seq_len
        self._init_fn = init_fn
        self.obs_memory = None
        self.memory_idx = 0

    def forward(self, input_obs: torch.Tensor, only_last_logit: bool = True, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Arguments:
            - input_obs (:obj:`torch.Tensor`): Input observation without sequence shape: (bs, *obs_shape)
            - only_last_logit (:obj:`bool`): if True 'logit' only contains the output corresponding to the current
            observation (shape: bs, embedding_dim), otherwise logit has shape (seq_len, bs, embedding_dim)
        Returns:
            - Dictionary containing the input_sequence 'input_seq' stored in memory and the transformer output 'logit'.
        """
        if self.obs_memory is None:
            self.reset_memory(torch.zeros_like(input_obs))  # init the memory with the size of the input observation
        assert self.obs_memory.shape[0] == self.seq_len
        # implements a fifo queue, self.memory_idx is index where to put the last element
        if self.memory_idx == self.seq_len-1:
            self.obs_memory = torch.roll(self.obs_memory, -1, 1)  # roll back of 1 position along dim 1 (sequence dim)
            self.obs_memory[self.memory_idx] = input_obs
        if self.memory_idx < self.seq_len:
            self.obs_memory[self.memory_idx] = input_obs
            if self.memory_idx != self.seq_len-1:
                self.memory_idx += 1
        out = self._model.forward(self.obs_memory, **kwargs)
        out['input_seq'] = self.obs_memory
        if only_last_logit:
            out['logit'] = out['logit'][self.memory_idx-1]
        return out

    def reset_memory(self, input_obs: torch.Tensor):
        init_obs = torch.zeros_like(input_obs)
        self.obs_memory = []  # List(bs, *obs_shape)
        for i in range(self.seq_len):
            self.obs_memory.append(init_obs.clone() if init_obs is not None else self._init_fn())
        self.obs_memory = default_collate(self.obs_memory)  # shape (N, bs, *obs_shape)
        self.memory_idx = 0

    def reset(self, *args, **kwargs):
        self.obs_memory = None
        self.memory_idx = 0
        if hasattr(self._model, 'reset'):
            return self._model.reset(*args, **kwargs)


def sample_action(logit=None, prob=None):
    if prob is None:
        prob = torch.softmax(logit, dim=-1)
    shape = prob.shape
    prob += 1e-8
    prob = prob.view(-1, shape[-1])
    # prob can also be treated as weight in multinomial sample
    action = torch.multinomial(prob, 1).squeeze(-1)
    action = action.view(*shape[:-1])
    return action


class ArgmaxSampleWrapper(IModelWrapper):
    r"""
    Overview:
        Used to help the model to sample argmax action
    """

    def forward(self, *args, **kwargs):
        output = self._model.forward(*args, **kwargs)
        assert isinstance(output, dict), "model output must be dict, but find {}".format(type(output))
        logit = output['logit']
        assert isinstance(logit, torch.Tensor) or isinstance(logit, list)
        if isinstance(logit, torch.Tensor):
            logit = [logit]
        if 'action_mask' in output:
            mask = output['action_mask']
            if isinstance(mask, torch.Tensor):
                mask = [mask]
            logit = [l.sub_(1e8 * (1 - m)) for l, m in zip(logit, mask)]
        action = [l.argmax(dim=-1) for l in logit]
        if len(action) == 1:
            action, logit = action[0], logit[0]
        output['action'] = action
        return output


class HybridArgmaxSampleWrapper(IModelWrapper):
    r"""
    Overview:
        Used to help the model to sample argmax action in hybrid action space,
        i.e.{'action_type': discrete, 'action_args', continuous}
    """

    def forward(self, *args, **kwargs):
        output = self._model.forward(*args, **kwargs)
        assert isinstance(output, dict), "model output must be dict, but find {}".format(type(output))
        if 'logit' not in output:
            return output
        logit = output['logit']
        assert isinstance(logit, torch.Tensor) or isinstance(logit, list)
        if isinstance(logit, torch.Tensor):
            logit = [logit]
        if 'action_mask' in output:
            mask = output['action_mask']
            if isinstance(mask, torch.Tensor):
                mask = [mask]
            logit = [l.sub_(1e8 * (1 - m)) for l, m in zip(logit, mask)]
        action = [l.argmax(dim=-1) for l in logit]
        if len(action) == 1:
            action, logit = action[0], logit[0]
        output = {'action': {'action_type': action, 'action_args': output['action_args']}, 'logit': logit}
        return output


class MultinomialSampleWrapper(IModelWrapper):
    r"""
    Overview:
        Used to helper the model get the corresponding action from the output['logits']
    Interfaces:
        register
    """

    def forward(self, *args, **kwargs):
        output = self._model.forward(*args, **kwargs)
        assert isinstance(output, dict), "model output must be dict, but find {}".format(type(output))
        logit = output['logit']
        assert isinstance(logit, torch.Tensor) or isinstance(logit, list)
        if isinstance(logit, torch.Tensor):
            logit = [logit]
        if 'action_mask' in output:
            mask = output['action_mask']
            if isinstance(mask, torch.Tensor):
                mask = [mask]
            logit = [l.sub_(1e8 * (1 - m)) for l, m in zip(logit, mask)]
        action = [sample_action(logit=l) for l in logit]
        if len(action) == 1:
            action, logit = action[0], logit[0]
        output['action'] = action
        return output


class EpsGreedySampleWrapper(IModelWrapper):
    r"""
    Overview:
        Epsilon greedy sampler used in collector_model to help balance exploration and exploitation.
    Interfaces:
        register
    """

    def forward(self, *args, **kwargs):
        eps = kwargs.pop('eps')
        output = self._model.forward(*args, **kwargs)
        assert isinstance(output, dict), "model output must be dict, but find {}".format(type(output))
        logit = output['logit']
        assert isinstance(logit, torch.Tensor) or isinstance(logit, list)
        if isinstance(logit, torch.Tensor):
            logit = [logit]
        if 'action_mask' in output:
            mask = output['action_mask']
            if isinstance(mask, torch.Tensor):
                mask = [mask]
            logit = [l.sub_(1e8 * (1 - m)) for l, m in zip(logit, mask)]
        else:
            mask = None
        action = []
        for i, l in enumerate(logit):
            if np.random.random() > eps:
                action.append(l.argmax(dim=-1))
            else:
                if mask:
                    action.append(sample_action(prob=mask[i].float()))
                else:
                    action.append(torch.randint(0, l.shape[-1], size=l.shape[:-1]))
        if len(action) == 1:
            action, logit = action[0], logit[0]
        output['action'] = action
        return output


class HybridEpsGreedySampleWrapper(IModelWrapper):
    r"""
    Overview:
        Epsilon greedy sampler used in collector_model to help balance exploration and exploitation.
        In hybrid action space, i.e.{'action_type': discrete, 'action_args', continuous}
    Interfaces:
        register, forward
    """

    def forward(self, *args, **kwargs):
        eps = kwargs.pop('eps')
        output = self._model.forward(*args, **kwargs)
        assert isinstance(output, dict), "model output must be dict, but find {}".format(type(output))
        logit = output['logit']
        assert isinstance(logit, torch.Tensor) or isinstance(logit, list)
        if isinstance(logit, torch.Tensor):
            logit = [logit]
        if 'action_mask' in output:
            mask = output['action_mask']
            if isinstance(mask, torch.Tensor):
                mask = [mask]
            logit = [l.sub_(1e8 * (1 - m)) for l, m in zip(logit, mask)]
        else:
            mask = None
        action = []
        for i, l in enumerate(logit):
            if np.random.random() > eps:
                action.append(l.argmax(dim=-1))
            else:
                if mask:
                    action.append(sample_action(prob=mask[i].float()))
                else:
                    action.append(torch.randint(0, l.shape[-1], size=l.shape[:-1]))
        if len(action) == 1:
            action, logit = action[0], logit[0]
        output = {'action': {'action_type': action, 'action_args': output['action_args']}, 'logit': logit}
        return output


class EpsGreedyMultinomialSampleWrapper(IModelWrapper):
    r"""
    Overview:
        Epsilon greedy sampler coupled with multinomial sample used in collector_model
        to help balance exploration and exploitation.
    Interfaces:
        register
    """

    def forward(self, *args, **kwargs):
        eps = kwargs.pop('eps')
        alpha = kwargs.pop('alpha')
        output = self._model.forward(*args, **kwargs)
        assert isinstance(output, dict), "model output must be dict, but find {}".format(type(output))
        logit = output['logit']
        assert isinstance(logit, torch.Tensor) or isinstance(logit, list)
        if isinstance(logit, torch.Tensor):
            logit = [logit]
        if 'action_mask' in output:
            mask = output['action_mask']
            if isinstance(mask, torch.Tensor):
                mask = [mask]
            logit = [l.sub_(1e8 * (1 - m)) for l, m in zip(logit, mask)]
        else:
            mask = None
        action = []
        for i, l in enumerate(logit):
            if np.random.random() > eps:
                prob = torch.softmax(output['logit'] / alpha, dim=-1)
                prob = prob / torch.sum(prob, 1, keepdims=True)
                pi_action = torch.zeros(prob.shape)
                pi_action = Categorical(prob)
                pi_action = pi_action.sample()
                action.append(pi_action)
            else:
                if mask:
                    action.append(sample_action(prob=mask[i].float()))
                else:
                    action.append(torch.randint(0, l.shape[-1], size=l.shape[:-1]))
        if len(action) == 1:
            action, logit = action[0], logit[0]
        output['action'] = action
        return output


class HybridEpsGreedyMultinomialSampleWrapper(IModelWrapper):
    """
    Overview:
        Epsilon greedy sampler coupled with multinomial sample used in collector_model
        to help balance exploration and exploitation.
        In hybrid action space, i.e.{'action_type': discrete, 'action_args', continuous}
    Interfaces:
        register
    """

    def forward(self, *args, **kwargs):
        eps = kwargs.pop('eps')
        output = self._model.forward(*args, **kwargs)
        assert isinstance(output, dict), "model output must be dict, but find {}".format(type(output))
        if 'logit' not in output:
            return output

        logit = output['logit']
        assert isinstance(logit, torch.Tensor) or isinstance(logit, list)
        if isinstance(logit, torch.Tensor):
            logit = [logit]
        if 'action_mask' in output:
            mask = output['action_mask']
            if isinstance(mask, torch.Tensor):
                mask = [mask]
            logit = [l.sub_(1e8 * (1 - m)) for l, m in zip(logit, mask)]
        else:
            mask = None
        action = []
        for i, l in enumerate(logit):
            if np.random.random() > eps:
                prob = torch.softmax(l, dim=-1)
                prob = prob / torch.sum(prob, 1, keepdims=True)
                pi_action = Categorical(prob)
                pi_action = pi_action.sample()
                action.append(pi_action)
            else:
                if mask:
                    action.append(sample_action(prob=mask[i].float()))
                else:
                    action.append(torch.randint(0, l.shape[-1], size=l.shape[:-1]))
        if len(action) == 1:
            action, logit = action[0], logit[0]
        output = {'action': {'action_type': action, 'action_args': output['action_args']}, 'logit': logit}
        return output


class EpsGreedySampleNGUWrapper(IModelWrapper):
    r"""
    Overview:
        eps is a dict n_env
        Epsilon greedy sampler used in collector_model to help balance exploratin and exploitation.
    Interfaces:
        register
    """

    def forward(self, *args, **kwargs):
        kwargs.pop('eps')
        eps = {i: 0.4 ** (1 + 8 * i / (args[0]['obs'].shape[0] - 1)) for i in range(args[0]['obs'].shape[0])}  # TODO
        output = self._model.forward(*args, **kwargs)
        assert isinstance(output, dict), "model output must be dict, but find {}".format(type(output))
        logit = output['logit']
        assert isinstance(logit, torch.Tensor) or isinstance(logit, list)
        if isinstance(logit, torch.Tensor):
            logit = [logit]
        if 'action_mask' in output:
            mask = output['action_mask']
            if isinstance(mask, torch.Tensor):
                mask = [mask]
            logit = [l.sub_(1e8 * (1 - m)) for l, m in zip(logit, mask)]
        else:
            mask = None
        action = []
        for i, l in enumerate(logit):
            if np.random.random() > eps[i]:
                action.append(l.argmax(dim=-1))
            else:
                if mask:
                    action.append(sample_action(prob=mask[i].float()))
                else:
                    action.append(torch.randint(0, l.shape[-1], size=l.shape[:-1]))
        if len(action) == 1:
            action, logit = action[0], logit[0]
        output['action'] = action
        return output


class EpsGreedySampleWrapperSql(IModelWrapper):
    r"""
    Overview:
        Epsilon greedy sampler coupled with multinomial sample used in collector_model
        to help balance exploration and exploitation.
    Interfaces:
        register
    """

    def forward(self, *args, **kwargs):
        eps = kwargs.pop('eps')
        alpha = kwargs.pop('alpha')
        output = self._model.forward(*args, **kwargs)
        assert isinstance(output, dict), "model output must be dict, but find {}".format(type(output))
        logit = output['logit']
        assert isinstance(logit, torch.Tensor) or isinstance(logit, list)
        if isinstance(logit, torch.Tensor):
            logit = [logit]
        if 'action_mask' in output:
            mask = output['action_mask']
            if isinstance(mask, torch.Tensor):
                mask = [mask]
            logit = [l.sub_(1e8 * (1 - m)) for l, m in zip(logit, mask)]
        else:
            mask = None
        action = []
        for i, l in enumerate(logit):
            if np.random.random() > eps:
                prob = torch.softmax(output['logit'] / alpha, dim=-1)
                prob = prob / torch.sum(prob, 1, keepdims=True)
                pi_action = torch.zeros(prob.shape)
                pi_action = Categorical(prob)
                pi_action = pi_action.sample()
                action.append(pi_action)
            else:
                if mask:
                    action.append(sample_action(prob=mask[i].float()))
                else:
                    action.append(torch.randint(0, l.shape[-1], size=l.shape[:-1]))
        if len(action) == 1:
            action, logit = action[0], logit[0]
        output['action'] = action
        return output


class ActionNoiseWrapper(IModelWrapper):
    r"""
    Overview:
        Add noise to collector's action output; Do clips on both generated noise and action after adding noise.
    Interfaces:
        register, __init__, add_noise, reset
    Arguments:
        - model (:obj:`Any`): Wrapped model class. Should contain ``forward`` method.
        - noise_type (:obj:`str`): The type of noise that should be generated, support ['gauss', 'ou'].
        - noise_kwargs (:obj:`dict`): Keyword args that should be used in noise init. Depends on ``noise_type``.
        - noise_range (:obj:`Optional[dict]`): Range of noise, used for clipping.
        - action_range (:obj:`Optional[dict]`): Range of action + noise, used for clip, default clip to [-1, 1].
    """

    def __init__(
            self,
            model: Any,
            noise_type: str = 'gauss',
            noise_kwargs: dict = {},
            noise_range: Optional[dict] = None,
            action_range: Optional[dict] = {
                'min': -1,
                'max': 1
            }
    ) -> None:
        super().__init__(model)
        self.noise_generator = create_noise_generator(noise_type, noise_kwargs)
        self.noise_range = noise_range
        self.action_range = action_range

    def forward(self, *args, **kwargs):
        output = self._model.forward(*args, **kwargs)
        assert isinstance(output, dict), "model output must be dict, but find {}".format(type(output))
        if 'action' in output or 'action_args' in output:
            key = 'action' if 'action' in output else 'action_args'
            action = output[key]
            assert isinstance(action, torch.Tensor)
            action = self.add_noise(action)
            output[key] = action
        return output

    def add_noise(self, action: torch.Tensor) -> torch.Tensor:
        r"""
        Overview:
            Generate noise and clip noise if needed. Add noise to action and clip action if needed.
        Arguments:
            - action (:obj:`torch.Tensor`): Model's action output.
        Returns:
            - noised_action (:obj:`torch.Tensor`): Action processed after adding noise and clipping.
        """
        noise = self.noise_generator(action.shape, action.device)
        if self.noise_range is not None:
            noise = noise.clamp(self.noise_range['min'], self.noise_range['max'])
        action += noise
        if self.action_range is not None:
            action = action.clamp(self.action_range['min'], self.action_range['max'])
        return action

    def reset(self) -> None:
        r"""
        Overview:
            Reset noise generator.
        """
        pass


class TargetNetworkWrapper(IModelWrapper):
    r"""
    Overview:
        Maintain and update the target network
    Interfaces:
        update, reset
    """

    def __init__(self, model: Any, update_type: str, update_kwargs: dict):
        super().__init__(model)
        assert update_type in ['momentum', 'assign']
        self._update_type = update_type
        self._update_kwargs = update_kwargs
        self._update_count = 0

    def reset(self, *args, **kwargs):
        target_update_count = kwargs.pop('target_update_count', None)
        self.reset_state(target_update_count)
        if hasattr(self._model, 'reset'):
            return self._model.reset(*args, **kwargs)

    def update(self, state_dict: dict, direct: bool = False) -> None:
        r"""
        Overview:
            Update the target network state dict

        Arguments:
            - state_dict (:obj:`dict`): the state_dict from learner model
            - direct (:obj:`bool`): whether to update the target network directly, \
                if true then will simply call the load_state_dict method of the model
        """
        if direct:
            self._model.load_state_dict(state_dict, strict=True)
            self._update_count = 0
        else:
            if self._update_type == 'assign':
                if (self._update_count + 1) % self._update_kwargs['freq'] == 0:
                    self._model.load_state_dict(state_dict, strict=True)
                self._update_count += 1
            elif self._update_type == 'momentum':
                theta = self._update_kwargs['theta']
                for name, p in self._model.named_parameters():
                    # default theta = 0.001
                    p.data = (1 - theta) * p.data + theta * state_dict[name]

    def reset_state(self, target_update_count: int = None) -> None:
        r"""
        Overview:
            Reset the update_count
        Arguments:
            target_update_count (:obj:`int`): reset target update count value.
        """
        if target_update_count is not None:
            self._update_count = target_update_count


class TeacherNetworkWrapper(IModelWrapper):
    r"""
    Overview:
        Set the teacher Network. Set the model's model.teacher_cfg to the input teacher_cfg

    Interfaces:
        register
    """

    def __init__(self, model, teacher_cfg):
        super().__init__(model)
        self._model._teacher_cfg = teacher_cfg


wrapper_name_map = {
    'base': BaseModelWrapper,
    'hidden_state': HiddenStateWrapper,
    'argmax_sample': ArgmaxSampleWrapper,
    'hybrid_argmax_sample': HybridArgmaxSampleWrapper,
    'eps_greedy_sample': EpsGreedySampleWrapper,
    'eps_greedy_sample_ngu': EpsGreedySampleNGUWrapper,
    'eps_greedy_sample_sql': EpsGreedySampleWrapperSql,
    'eps_greedy_multinomial_sample': EpsGreedyMultinomialSampleWrapper,
    'hybrid_eps_greedy_sample': HybridEpsGreedySampleWrapper,
    'hybrid_eps_greedy_multinomial_sample': HybridEpsGreedyMultinomialSampleWrapper,
    'multinomial_sample': MultinomialSampleWrapper,
    'action_noise': ActionNoiseWrapper,
    'transformer': TransformerWrapper,
    # model wrapper
    'target': TargetNetworkWrapper,
    'teacher': TeacherNetworkWrapper,
}


def model_wrap(model, wrapper_name: str = None, **kwargs):
    if wrapper_name in wrapper_name_map:
        if not isinstance(model, IModelWrapper):
            model = wrapper_name_map['base'](model)
        model = wrapper_name_map[wrapper_name](model, **kwargs)
    return model


def register_wrapper(name: str, wrapper_type: type):
    r"""
    Overview:
        Register new wrapper to wrapper_name_map
    Arguments:
        - name (:obj:`str`): the name of the wrapper
        - wrapper_type (subclass of :obj:`IModelWrapper`): the wrapper class added to the plguin_name_map
    """
    assert isinstance(name, str)
    assert issubclass(wrapper_type, IModelWrapper)
    wrapper_name_map[name] = wrapper_type
