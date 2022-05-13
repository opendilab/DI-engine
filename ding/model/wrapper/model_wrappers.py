from typing import Any, Tuple, Callable, Optional, List, Dict
from abc import ABC

import numpy as np
import torch
from ding.torch_utils import get_tensor_data
from ding.rl_utils import create_noise_generator
from torch.distributions import Categorical, Independent, Normal
from ding.utils.data import default_collate
import torch.nn.functional as F


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


def zeros_like(h):
    if isinstance(h, torch.Tensor):
        return torch.zeros_like(h)
    elif isinstance(h, (list, tuple)):
        return [zeros_like(t) for t in h]
    elif isinstance(h, dict):
        return {k: zeros_like(v) for k, v in h.items()}
    else:
        raise TypeError("not support type: {}".format(h))


class HiddenStateWrapper(IModelWrapper):

    def __init__(
            self,
            model: Any,
            state_num: int,
            save_prev_state: bool = False,
            init_fn: Callable = lambda: None,
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
        # This is to maintain hidden states （when it comes to this wrapper, \
        # map self._state into data['prev_value] and update next_state, store in self._state)
        self._state = {i: init_fn() for i in range(state_num)}
        self._save_prev_state = save_prev_state
        self._init_fn = init_fn

    def forward(self, data, **kwargs):
        state_id = kwargs.pop('data_id', None)
        valid_id = kwargs.pop('valid_id', None)  # None, not used in any code in DI-engine
        data, state_info = self.before_forward(data, state_id)  # update data['prev_state'] with self._state
        output = self._model.forward(data, **kwargs)
        h = output.pop('next_state', None)
        if h is not None:
            self.after_forward(h, state_info, valid_id)  # this is to store the 'next hidden state' for each time step
        if self._save_prev_state:
            prev_state = get_tensor_data(data['prev_state'])
            # for compatibility, because of the incompatibility between None and torch.Tensor
            for i in range(len(prev_state)):
                if prev_state[i] is None:
                    prev_state[i] = zeros_like(h[0])
            output['prev_state'] = prev_state
        return output

    def reset(self, *args, **kwargs):
        state = kwargs.pop('state', None)
        state_id = kwargs.get('data_id', None)
        self.reset_state(state, state_id)
        if hasattr(self._model, 'reset'):
            return self._model.reset(*args, **kwargs)

    def reset_state(self, state: Optional[list] = None, state_id: Optional[list] = None) -> None:
        if state_id is None:  # train: init all states
            state_id = [i for i in range(self._state_num)]
        if state is None:  # collect: init state that are done
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


class TransformerInputWrapper(IModelWrapper):

    def __init__(self, model: Any, seq_len: int, init_fn: Callable = lambda: None) -> None:
        """
        Overview:
            Given N the length of the sequences received by a Transformer model, maintain the last N-1 input
            observations. In this way we can provide at each step all the observations needed by Transformer to
            compute its output. We need this because some methods such as 'collect' and 'evaluate' only provide the
            model 1 observation per step and don't have memory of past observations, but Transformer needs a sequence
            of N observations. The wrapper method ``forward`` will save the input observation in a FIFO memory of
            length N and the method ``reset`` will reset the memory. The empty memory spaces will be initialized
            with 'init_fn' or zero by calling the method ``reset_input``. Since different env can terminate at
            different steps, the method ``reset_memory_entry`` only initializes the memory of specific environments in
            the batch size.
        Arguments:
            - model (:obj:`Any`): Wrapped model class, should contain forward method.
            - seq_len (:obj:`int`): Number of past observations to remember.
            - init_fn (:obj:`Callable`): The function which is used to init every memory locations when init and reset.
        """
        super().__init__(model)
        self.seq_len = seq_len
        self._init_fn = init_fn
        self.obs_memory = None  # shape (N, bs, *obs_shape)
        self.init_obs = None  # sample of observation used to initialize the memory
        self.bs = None
        self.memory_idx = []  # len bs, index of where to put the next element in the sequence for each batch

    def forward(self,
                input_obs: torch.Tensor,
                only_last_logit: bool = True,
                data_id: List = None,
                **kwargs) -> Dict[str, torch.Tensor]:
        """
        Arguments:
            - input_obs (:obj:`torch.Tensor`): Input observation without sequence shape: (bs, *obs_shape)
            - only_last_logit (:obj:`bool`): if True 'logit' only contains the output corresponding to the current
                observation (shape: bs, embedding_dim), otherwise logit has shape (seq_len, bs, embedding_dim)
            - data_id (:obj:`List`): id of the envs that are currently running. Memory update and logits return has only
                effect for those environments. If `None` it is considered that all envs are running.
        Returns:
            - Dictionary containing the input_sequence 'input_seq' stored in memory and the transformer output 'logit'.
        """
        if self.obs_memory is None:
            self.reset_input(torch.zeros_like(input_obs))  # init the memory with the size of the input observation
        if data_id is None:
            data_id = list(range(self.bs))
        assert self.obs_memory.shape[0] == self.seq_len
        # implements a fifo queue, self.memory_idx is index where to put the last element
        for i, b in enumerate(data_id):
            if self.memory_idx[b] == self.seq_len:
                # roll back of 1 position along dim 1 (sequence dim)
                self.obs_memory[:, b] = torch.roll(self.obs_memory[:, b], -1, 0)
                self.obs_memory[self.memory_idx[b] - 1, b] = input_obs[i]
            if self.memory_idx[b] < self.seq_len:
                self.obs_memory[self.memory_idx[b], b] = input_obs[i]
                if self.memory_idx != self.seq_len:
                    self.memory_idx[b] += 1
        out = self._model.forward(self.obs_memory, **kwargs)
        out['input_seq'] = self.obs_memory
        if only_last_logit:
            # return only the logits for running environments
            out['logit'] = [out['logit'][self.memory_idx[b] - 1][b] for b in range(self.bs) if b in data_id]
            out['logit'] = default_collate(out['logit'])
        return out

    def reset_input(self, input_obs: torch.Tensor):
        """
        Overview:
            Initialize the whole memory
        """
        init_obs = torch.zeros_like(input_obs)
        self.init_obs = init_obs
        self.obs_memory = []  # List(bs, *obs_shape)
        for i in range(self.seq_len):
            self.obs_memory.append(init_obs.clone() if init_obs is not None else self._init_fn())
        self.obs_memory = default_collate(self.obs_memory)  # shape (N, bs, *obs_shape)
        self.bs = self.init_obs.shape[0]
        self.memory_idx = [0 for _ in range(self.bs)]

    # called before evaluation
    # called after each evaluation iteration for each done env
    # called after each collect iteration for each done env
    def reset(self, *args, **kwargs):
        state_id = kwargs.get('data_id', None)
        input_obs = kwargs.get('input_obs', None)
        if input_obs is not None:
            self.reset_input(input_obs)
        if state_id is not None:
            self.reset_memory_entry(state_id)
        if input_obs is None and state_id is None:
            self.obs_memory = None
        if hasattr(self._model, 'reset'):
            return self._model.reset(*args, **kwargs)

    def reset_memory_entry(self, state_id: Optional[list] = None) -> None:
        """
        Overview:
            Reset specific batch of the memory, batch ids are specified in 'state_id'
        """
        assert self.init_obs is not None, 'Call method "reset_memory" first'
        for _id in state_id:
            self.memory_idx[_id] = 0
            self.obs_memory[:, _id] = self.init_obs[_id]  # init the corresponding sequence with broadcasting


class TransformerSegmentWrapper(IModelWrapper):

    def __init__(self, model: Any, seq_len: int) -> None:
        """
        Overview:
            Given T the length of a trajectory and N the length of the sequences received by a Transformer model,
            split T in sequences of N elements and forward each sequence one by one. If T % N != 0, the last sequence
            will be zero-padded. Usually used during Transformer training phase.
        Arguments:
            - model (:obj:`Any`): Wrapped model class, should contain forward method.
            - seq_len (:obj:`int`): N, length of a sequence.
        """
        super().__init__(model)
        self.seq_len = seq_len

    def forward(self, obs: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Arguments:
            - data (:obj:`dict`): Dict type data, including at least \
                ['main_obs', 'target_obs', 'action', 'reward', 'done', 'weight']
        Returns:
            - List containing a dict of the model output for each sequence.
        """
        sequences = list(torch.split(obs, self.seq_len, dim=0))
        if sequences[-1].shape[0] < self.seq_len:
            last = sequences[-1].clone()
            diff = self.seq_len - last.shape[0]
            sequences[-1] = F.pad(input=last, pad=(0, 0, 0, 0, 0, diff), mode='constant', value=0)
        outputs = []
        for i, seq in enumerate(sequences):
            out = self._model.forward(seq, **kwargs)
            outputs.append(out)
        out = {}
        for k in outputs[0].keys():
            out_k = [o[k] for o in outputs]
            out_k = torch.cat(out_k, dim=0)
            out[k] = out_k
        return out


class TransformerMemoryWrapper(IModelWrapper):

    def __init__(
            self,
            model: Any,
            batch_size: int,
    ) -> None:
        """
        Overview:
            Stores a copy of the Transformer memory in order to be reused across different phases. To make it more
             clear, suppose the training pipeline is divided into 3 phases: evaluate, collect, learn. The goal of the
             wrapper is to maintain the content of the memory at the end of each phase and reuse it when the same phase
             is executed again. In this way, it prevents different phases to interferer each other memory.
        Arguments:
            - model (:obj:`Any`): Wrapped model class, should contain forward method.
            - batch_size (:obj:`int`): Memory batch size.
        """
        super().__init__(model)
        # shape (layer_num, memory_len, bs, embedding_dim)
        self._model.reset_memory(batch_size=batch_size)
        self.memory = self._model.get_memory()
        self.mem_shape = self.memory.shape

    def forward(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Arguments:
            - data (:obj:`dict`): Dict type data, including at least \
                ['main_obs', 'target_obs', 'action', 'reward', 'done', 'weight']
        Returns:
            - Output of the forward method.
        """
        self._model.reset_memory(state=self.memory)
        out = self._model.forward(*args, **kwargs)
        self.memory = self._model.get_memory()
        return out

    def reset(self, *args, **kwargs):
        state_id = kwargs.get('data_id', None)
        if state_id is None:
            self.memory = torch.zeros(self.mem_shape)
        else:
            self.reset_memory_entry(state_id)
        if hasattr(self._model, 'reset'):
            return self._model.reset(*args, **kwargs)

    def reset_memory_entry(self, state_id: Optional[list] = None) -> None:
        """
        Overview:
            Reset specific batch of the memory, batch ids are specified in 'state_id'
        """
        for _id in state_id:
            self.memory[:, :, _id] = torch.zeros((self.mem_shape[-1]))

    def show_memory_occupancy(self, layer=0) -> None:
        memory = self.memory
        memory_shape = memory.shape
        print('Layer {}-------------------------------------------'.format(layer))
        for b in range(memory_shape[-2]):
            print('b{}: '.format(b), end='')
            for m in range(memory_shape[1]):
                if sum(abs(memory[layer][m][b].flatten())) != 0:
                    print(1, end='')
                else:
                    print(0, end='')
            print()


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
        Used to help the model get the corresponding action from the output['logits']
    Interfaces:
        register
    """

    def forward(self, *args, **kwargs):
        if 'alpha' in kwargs.keys():
            alpha = kwargs.pop('alpha')
        else:
            alpha = None
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
        if alpha is None:
            action = [sample_action(logit=l) for l in logit]
        else:
            # Note that if alpha is passed in here, we will divide logit by alpha.
            action = [sample_action(logit=l / alpha) for l in logit]
        if len(action) == 1:
            action, logit = action[0], logit[0]
        output['action'] = action
        return output


class EpsGreedySampleWrapper(IModelWrapper):
    r"""
    Overview:
        Epsilon greedy sampler used in collector_model to help balance exploratin and exploitation.
        The type of eps can vary from different algorithms, such as:
        - float (i.e. python native scalar): for almost normal case
        - Dict[str, float]: for algorithm NGU
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
        if isinstance(eps, dict):
            # for NGU policy, eps is a dict, each collect env has a different eps
            for i, l in enumerate(logit[0]):
                eps_tmp = eps[i]
                if np.random.random() > eps_tmp:
                    action.append(l.argmax(dim=-1))
                else:
                    if mask is not None:
                        action.append(
                            sample_action(prob=mask[0][i].float().unsqueeze(0)).to(logit[0].device).squeeze(0)
                        )
                    else:
                        action.append(torch.randint(0, l.shape[-1], size=l.shape[:-1]).to(logit[0].device))
            action = torch.stack(action, dim=-1)  # shape torch.size([env_num])
        else:
            for i, l in enumerate(logit):
                if np.random.random() > eps:
                    action.append(l.argmax(dim=-1))
                else:
                    if mask is not None:
                        action.append(sample_action(prob=mask[i].float()))
                    else:
                        action.append(torch.randint(0, l.shape[-1], size=l.shape[:-1]))
            if len(action) == 1:
                action, logit = action[0], logit[0]
        output['action'] = action
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
        if 'alpha' in kwargs.keys():
            alpha = kwargs.pop('alpha')
        else:
            alpha = None
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
                if alpha is None:
                    action = [sample_action(logit=l) for l in logit]
                else:
                    # Note that if alpha is passed in here, we will divide logit by alpha.
                    action = [sample_action(logit=l / alpha) for l in logit]
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
                action = [sample_action(logit=l) for l in logit]
            else:
                if mask:
                    action.append(sample_action(prob=mask[i].float()))
                else:
                    action.append(torch.randint(0, l.shape[-1], size=l.shape[:-1]))
        if len(action) == 1:
            action, logit = action[0], logit[0]
        output = {'action': {'action_type': action, 'action_args': output['action_args']}, 'logit': logit}
        return output


class HybridReparamMultinomialSampleWrapper(IModelWrapper):
    """
    Overview:
        Reparameterization sampler coupled with multinomial sample used in collector_model
        to help balance exploration and exploitation.
        In hybrid action space, i.e.{'action_type': discrete, 'action_args', continuous}
    Interfaces:
        forward
    """

    def forward(self, *args, **kwargs):
        output = self._model.forward(*args, **kwargs)
        assert isinstance(output, dict), "model output must be dict, but find {}".format(type(output))

        logit = output['logit']  # logit: {'action_type': action_type_logit, 'action_args': action_args_logit}
        # discrete part
        action_type_logit = logit['action_type']
        prob = torch.softmax(action_type_logit, dim=-1)
        pi_action = Categorical(prob)
        action_type = pi_action.sample()
        # continuous part
        mu, sigma = logit['action_args']['mu'], logit['action_args']['sigma']
        dist = Independent(Normal(mu, sigma), 1)
        action_args = dist.sample()
        action = {'action_type': action_type, 'action_args': action_args}
        output['action'] = action
        return output


class HybridDeterministicArgmaxSampleWrapper(IModelWrapper):
    """
    Overview:
        Deterministic sampler coupled with argmax sample used in eval_model.
        In hybrid action space, i.e.{'action_type': discrete, 'action_args', continuous}
    Interfaces:
        forward
    """

    def forward(self, *args, **kwargs):
        output = self._model.forward(*args, **kwargs)
        assert isinstance(output, dict), "model output must be dict, but find {}".format(type(output))
        logit = output['logit']  # logit: {'action_type': action_type_logit, 'action_args': action_args_logit}
        # discrete part
        action_type_logit = logit['action_type']
        action_type = action_type_logit.argmax(dim=-1)
        # continuous part
        mu = logit['action_args']['mu']
        action_args = mu
        action = {'action_type': action_type, 'action_args': action_args}
        output['action'] = action
        return output


class DeterministicSample(IModelWrapper):
    """
    Overview:
        Deterministic sampler (just use mu directly) used in eval_model.
    Interfaces:
        forward
    """

    def forward(self, *args, **kwargs):
        output = self._model.forward(*args, **kwargs)
        assert isinstance(output, dict), "model output must be dict, but find {}".format(type(output))
        output['action'] = output['logit']['mu']
        return output


class ReparamSample(IModelWrapper):
    """
    Overview:
        Reparameterization gaussian sampler used in collector_model.
    Interfaces:
        forward
    """

    def forward(self, *args, **kwargs):
        output = self._model.forward(*args, **kwargs)
        assert isinstance(output, dict), "model output must be dict, but find {}".format(type(output))
        mu, sigma = output['logit']['mu'], output['logit']['sigma']
        dist = Independent(Normal(mu, sigma), 1)
        output['action'] = dist.sample()
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
    'eps_greedy_multinomial_sample': EpsGreedyMultinomialSampleWrapper,
    'deterministic_sample': DeterministicSample,
    'reparam_sample': ReparamSample,
    'hybrid_eps_greedy_sample': HybridEpsGreedySampleWrapper,
    'hybrid_eps_greedy_multinomial_sample': HybridEpsGreedyMultinomialSampleWrapper,
    'hybrid_reparam_multinomial_sample': HybridReparamMultinomialSampleWrapper,
    'hybrid_deterministic_argmax_sample': HybridDeterministicArgmaxSampleWrapper,
    'multinomial_sample': MultinomialSampleWrapper,
    'action_noise': ActionNoiseWrapper,
    'transformer_input': TransformerInputWrapper,
    'transformer_segment': TransformerSegmentWrapper,
    'transformer_memory': TransformerMemoryWrapper,
    # model wrapper
    'target': TargetNetworkWrapper,
    'teacher': TeacherNetworkWrapper,
}


def model_wrap(model, wrapper_name: str = None, **kwargs):
    if wrapper_name in wrapper_name_map:
        if not isinstance(model, IModelWrapper):
            model = wrapper_name_map['base'](model)
        model = wrapper_name_map[wrapper_name](model, **kwargs)
    else:
        raise TypeError("not support model_wrapper type: {}".format(wrapper_name))
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
