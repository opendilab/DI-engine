import copy
from abc import ABC, abstractmethod, abstractclassmethod
from collections import OrderedDict
from typing import Any, Tuple, Callable, Union, Optional, Dict

import numpy as np
import torch
from nervex.torch_utils import get_tensor_data
from nervex.rl_utils import create_noise_generator


class IAgentPlugin(ABC):
    r"""
    Overview:
        the base class of Agent Plugins

    Interfaces:
        register
    """

    @abstractclassmethod
    def register(cls: type, agent: Any, **kwargs) -> None:
        r"""
        Overview:
            the register function that every subclass of IAgentPlugin should implement
        """
        """inplace modify agent"""
        raise NotImplementedError


IAgentStatelessPlugin = IAgentPlugin


class IAgentStatefulPlugin(IAgentPlugin):
    r"""
    Overview:
        the base class of Agent Plugins that requires to store states
    Interfaces:
        __init__, reset
    """

    @abstractmethod
    def __init__(self, *args, **kwargs) -> None:
        r"""
        Overview
            the init function that the Agent Plugins with states should implement
            used to init the stored states
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        r"""
        Overview
            the reset function that the Agent Plugins with states should implement
            used to reset the stored states
        """
        raise NotImplementedError


class GradHelper(IAgentStatelessPlugin):
    r"""
    Overview:
        GradHelper help the agent to enable grad or disable grad while calling forward method

    Interfaces:
        register

    Examples:
        >>> GradHelper.register(actor_agent, Flase)
        >>> GradHelper.register(learner_agent, True)
    """

    @classmethod
    def register(cls: type, agent: Any, enable_grad: bool) -> None:
        r"""
        Overview:
            after register, the agent.foward method with be set to enable grad or disable grad

        Arguments:
            - agent (:obj:`Any`): the wrapped agent class, should contain forward methods
            - enbale_grad (:obj:`bool`): whether to enable grad or disable grad while forwarding.
        """

        def grad_wrapper(fn):
            context = torch.enable_grad() if enable_grad else torch.no_grad()

            def wrapper(*args, **kwargs):
                with context:
                    return fn(*args, **kwargs)

            return wrapper

        agent.forward = grad_wrapper(agent.forward)


class HiddenStateHelper(IAgentStatefulPlugin):
    """
    Overview:
        maintain the hidden state for RNN-base model, each sample in a batch has its own state

    Interfaces:
        register

    .. note::
        1. this helper must deal with a actual batch with some parts of samples(e.g: 6 samples of state_num 8)
        2. this helper must deal with the single sample state reset
    """

    @classmethod
    def register(
            cls: type,
            agent: Any,
            state_num: int,
            save_prev_state: bool = False,
            init_fn: Callable = lambda: None
    ) -> None:
        r"""
        Overview:
            init the maintain state and state function, then wrap the agent.foward nethod with auto saved data\
                ['prev_state'] input, and create the agent.reset method the HiddenState

        Arguments:
            - agent(:obj:`Any`): the wrapped agent class, should contain forward methods
            - state_num (:obj:`int`): the num of states to process
            - save_prev_state (:obj:`bool`): whether to output the prev state in output['prev_state']
            - init_fn (:obj:`Callable`): the init function to init every hidden state when init and reset,\
                default set to return None for hidden states
        """
        state_manager = cls(state_num, init_fn=init_fn)
        agent._state_manager = state_manager

        def forward_state_wrapper(forward_fn):

            def wrapper(data, **kwargs):
                state_id = kwargs.pop('data_id', None)
                data, state_info = agent._state_manager.before_forward(data, state_id)
                output = forward_fn(data, **kwargs)
                h = output.pop('next_state', None)
                if h:
                    agent._state_manager.after_forward(h, state_info)
                if save_prev_state:
                    prev_state = get_tensor_data(data['prev_state'])
                    output['prev_state'] = prev_state
                return output

            return wrapper

        def reset_state_wrapper(reset_fn):

            def wrapper(*args, **kwargs):
                state = kwargs.pop('state', None)
                state_id = kwargs.get('data_id', None)
                agent._state_manager.reset(state, state_id)
                return reset_fn(*args, **kwargs)

            return wrapper

        agent.forward = forward_state_wrapper(agent.forward)
        agent.reset = reset_state_wrapper(agent.reset)

    def __init__(self, state_num: int, init_fn: Callable) -> None:
        self._state_num = state_num
        self._state = {i: init_fn() for i in range(state_num)}
        self._init_fn = init_fn

    def reset(self, state: Optional[list] = None, state_id: Optional[list] = None) -> None:
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

    def after_forward(self, h: Any, state_info: dict) -> None:
        assert len(h) == len(state_info), '{}/{}'.format(len(h), len(state_info))
        for i, idx in enumerate(state_info.keys()):
            self._state[idx] = h[i]


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


class ArgmaxSampleHelper(IAgentStatelessPlugin):
    r"""
    Overview:
        Used to help the agent to sample argmax action

    Interfaces:
        register

    Examples:
        >>> ArgmaxSampleHelper.register(actor_agent)
    """

    @classmethod
    def register(cls: type, agent: Any) -> None:
        r"""
        Overview:
            wrap the agent.forward method with argmax output['action']

        Arguments:
            - agent (:obj:`Any`): the wrapped agent class, should contain forward methods
        """

        def sample_wrapper(forward_fn: Callable) -> Callable:

            def wrapper(*args, **kwargs):
                output = forward_fn(*args, **kwargs)
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

            return wrapper

        agent.forward = sample_wrapper(agent.forward)


class MultinomialSampleHelper(IAgentStatelessPlugin):
    r"""
    Overview:
        Used to helper the agent get the corresponding action from the output['logits']

    Interfaces:
        register
    """

    @classmethod
    def register(cls: type, agent: Any) -> None:
        r"""
        Overview:
            auto wrap the agent.forward method and take the output['logit'] as probs of action to calculate\
                the output['action']

        Arguments:
            - agent (:obj:`Any`): the wrapped agent class, should contain forward methods
        """

        def sample_wrapper(forward_fn: Callable) -> Callable:

            def wrapper(*args, **kwargs):
                output = forward_fn(*args, **kwargs)
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

            return wrapper

        agent.forward = sample_wrapper(agent.forward)


class EpsGreedySampleHelper(IAgentStatelessPlugin):
    r"""
    Overview:
        the eps greedy sampler used in actor_agent to help balance exploratin and exploitation

    Interfaces:
        register
    """

    @classmethod
    def register(cls: type, agent: Any) -> None:
        r"""
        Overview:
            auto wrap the agent.forward method with eps prob to take a random action

        Arguments:
            - agent (:obj:`Any`): the wrapped agent class, should contain forward methods

        .. note::
            after wrapped by the EpsGreedySampleHelper, the agent.forward should take **kwargs of {'eps': float}

        """

        def sample_wrapper(forward_fn: Callable) -> Callable:

            def wrapper(*args, **kwargs):
                eps = kwargs.pop('eps')
                output = forward_fn(*args, **kwargs)
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

            return wrapper

        agent.forward = sample_wrapper(agent.forward)


class ActionNoiseHelper(IAgentStatefulPlugin):

    @classmethod
    def register(
            cls: type,
            agent: Any,
            noise_type: str = 'gauss',
            noise_kwargs: dict = {},
            noise_range: Optional[dict] = None,
            action_range: Optional[dict] = {
                'min': -1,
                'max': 1
            }
    ) -> None:
        noise_helper = cls(noise_type, noise_kwargs, noise_range, action_range)
        agent._noise_helper = noise_helper

        def noise_wrapper(forward_fn: Callable) -> Callable:

            def wrapper(*args, **kwargs):
                output = forward_fn(*args, **kwargs)
                assert isinstance(output, dict), "model output must be dict, but find {}".format(type(output))
                if 'action' in output:
                    action = output['action']
                    assert isinstance(action, torch.Tensor)
                    action = agent._noise_helper.add_noise(action)
                    output['action'] = action
                return output

            return wrapper

        agent.forward = noise_wrapper(agent.forward)

    def __init__(
            self,
            noise_type: str = 'gauss',
            noise_kwargs: dict = {},
            noise_range: Optional[dict] = None,
            action_range: Optional[dict] = {
                'min': -1,
                'max': 1
            },
    ) -> None:
        self.noise_generator = create_noise_generator(noise_type, noise_kwargs)
        self.noise_range = noise_range
        self.action_range = action_range

    def add_noise(self, action: torch.Tensor) -> torch.Tensor:
        noise = self.noise_generator(action.shape, action.device)
        if self.noise_range is not None:
            noise = noise.clamp(self.noise_range['min'], self.noise_range['max'])
        action += noise
        if self.action_range is not None:
            action = action.clamp(self.action_range['min'], self.action_range['max'])
        return action

    def reset(self):
        pass


class TargetNetworkHelper(IAgentStatefulPlugin):
    r"""
    Overview:
        Maintain and update the target network

    Interfaces:
        register, update, reset
    """

    @classmethod
    def register(cls: type, agent: Any, update_type: str, update_kwargs: dict):
        r"""
        Overview:
            help maintain the target network, including reset the target when the wrapped agent reset,\
                create the agent.update method with the update method of cls class i.e TargetNetworkHelper\
                    class itself

        Arguments:
            - agent (:obj:`Any`): the wrapped agent class, should contain forward methods
            - update_type (:obj:`str`): the update_type used to update the momentum network, support \
                ['momentum', 'assign']
            - update_kwargs (:obj:`dict`): the update kwargs
        """
        target_network = cls(agent.model, update_type, update_kwargs)
        agent._target_network = target_network

        def reset_wrapper(reset_fn):

            def wrapper(*args, **kwargs):
                agent._target_network.reset()
                return reset_fn(*args, **kwargs)

            return wrapper

        setattr(agent, 'update', getattr(agent._target_network, 'update'))
        agent.reset = reset_wrapper(agent.reset)

    def __init__(self, model: torch.nn.Module, update_type: str, kwargs: dict) -> None:
        self._model = model
        assert update_type in ['momentum', 'assign']
        self._update_type = update_type
        self._update_kwargs = kwargs
        self._update_count = 0

    def update(self, state_dict: dict, direct: bool = False) -> None:
        r"""
        Overview:
            update the target network state dict

        Arguments:
            - state_dict (:obj:`dict`): the state_dict from learner agent
            - direct (:obj:`bool`): whether to update the target network directly, \
                if ture then will simply call the load_state_dict method of the model
        """
        if direct:
            self._model.load_state_dict(state_dict, strict=True)
            self._update_count = 0
        elif self._update_type == 'assign':
            if (self._update_count + 1) % self._update_kwargs['freq'] == 0:
                self._model.load_state_dict(state_dict, strict=True)
            self._update_count += 1
        elif self._update_type == 'momentum':
            theta = self._update_kwargs['theta']
            for name, p in self._model.named_parameters():
                # default theta = 0.001
                p.data = (1 - theta) * p.data + theta * state_dict[name]

    def reset(self) -> None:
        r"""
        Overview:
            reset the update_count
        """
        self._update_count = 0


class TeacherNetworkHelper(IAgentStatelessPlugin):
    r"""
    Overview:
        set the teacher Network

    Interfaces:
        register
    """

    @classmethod
    def register(cls: type, agent: Any, teacher_cfg: dict) -> None:
        r"""
        Overview:
            set the agent's agent.teacher_cfg to the input teacher_cfg

        Arguments:
            - agent (:obj:`Any`): the registered agent
        """
        agent._teacher_cfg = teacher_cfg


plugin_name_map = {
    'grad': GradHelper,
    'hidden_state': HiddenStateHelper,
    'argmax_sample': ArgmaxSampleHelper,
    'eps_greedy_sample': EpsGreedySampleHelper,
    'multinomial_sample': MultinomialSampleHelper,
    'action_noise': ActionNoiseHelper,
    # model plugin
    'target': TargetNetworkHelper,
    'teacher': TeacherNetworkHelper,
}


def add_plugin(agent: 'BaseAgent', plugin_name: str, **kwargs) -> None:  # noqa
    r"""
    Overview:
        add plugin with plugin_name and kwargs to agent
    Arguments:
        - agent (:obj:`Any`): the agent to register plugin to
        - plugin_name (:obj:`str`): agent plugin name, which must be in plugin_name_map
    """
    if plugin_name not in plugin_name_map:
        raise KeyError("invalid agent plugin name: {}".format(plugin_name))
    else:
        plugin_name_map[plugin_name].register(agent, **kwargs)


def register_plugin(name: str, plugin_type: type):
    r"""
    Overview:
        register new plugin to plugin_name_map
    Arguments:
        - name (:obj:`str`): the name of the plugin
        - plugin_type (subclass of :obj:`IAgentPlugin`): the plugin class added to the plguin_name_map
    """
    assert isinstance(name, str)
    assert issubclass(plugin_type, IAgentPlugin)
    plugin_name_map[name] = plugin_type
