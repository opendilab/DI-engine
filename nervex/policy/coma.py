from typing import List, Dict, Any, Tuple, Union, Optional
from collections import namedtuple, deque
import torch
import copy
from easydict import EasyDict

from nervex.torch_utils import Adam, to_device
from nervex.data import default_collate, default_decollate
from nervex.rl_utils import coma_data, coma_error, get_epsilon_greedy_fn, Adder
from nervex.model import ComaNetwork, model_wrap
from nervex.data import timestep_collate
from nervex.utils import POLICY_REGISTRY
from .base_policy import Policy


@POLICY_REGISTRY.register('coma')
class COMAPolicy(Policy):
    config = dict(
        # (str) RL policy register name (refer to function "POLICY_REGISTRY").
        type='coma',
        # (bool) Whether to use cuda for network.
        cuda=False,
        # (bool) Whether to use multi gpu
        multi_gpu=False,
        # (bool) Whether the RL algorithm is on-policy or off-policy.
        on_policy=True,
        # (bool) Whether use priority(priority sample, IS weight, update priority)
        priority=False,
        learn=dict(
            update_per_collect=1,
            batch_size=32,
            learning_rate=0.0005,
            weight_decay=0.00001,
            # ==============================================================
            # The following configs is algorithm-specific
            # ==============================================================
            # (float) target network update weight, theta * new_w + (1 - theta) * old_w, defaults in [0, 0.1]
            target_update_theta=0.001,
            # (float) discount factor for future reward, defaults int [0, 1]
            discount_factor=0.99,
            # (float) the trade-off factor of td-lambda, which balances 1step td and mc(nstep td in practice)
            td_lambda=0.8,
            # (float) the loss weight of value network, policy network weight is set to 1
            value_weight=1.0,
            # (float) the loss weight of entropy regularization, policy network weight is set to 1
            entropy_weight=0.01,
        ),
        collect=dict(
            # (int) collect n_episode data, train model n_iteration time
            n_episode=6,
            # (int) unroll length of a train iteration(gradient update step)
            unroll_len=16,
        ),
        eval=dict(),
        other=dict(
            eps=dict(
                type='exp',
                start=0.5,
                end=0.01,
                decay=100000,
            ),
            replay_buffer=dict(
                # (int) max size of replay buffer
                replay_buffer_size=64,
                # (int) max use count of data, if count is bigger than this value, the data will be removed from buffer
                max_use=100,
            ),
        ),
    )

    def _init_learn(self) -> None:
        """
        Overview:
            Init the learner model of COMAPolicy

        Arguments:
            .. note::

                The _init_learn method takes the argument from the self._cfg.learn in the config file

            - learning_rate (:obj:`float`): The learning rate fo the optimizer
            - gamma (:obj:`float`): The discount factor
            - lambda (:obj:`float`): The lambda factor, determining the mix of bootstrapping\
                vs further accumulation of multistep returns at each timestep,
            - value_wight(:obj:`float`): The weight of value loss in total loss
            - entropy_weight(:obj:`float`): The weight of entropy loss in total loss
            - agent_num (:obj:`int`): Since this is a multi-agent algorithm, we need to input the agent num.
            - batch_size (:obj:`int`): Need batch size info to init hidden_state plugins
        """
        self._priority = self._cfg.priority
        assert not self._priority, "not implemented priority in PPO"
        self._optimizer = Adam(self._model.parameters(), lr=self._cfg.learn.learning_rate)
        algo_cfg = self._cfg.learn.algo
        self._gamma = algo_cfg.discount_factor
        self._lambda = algo_cfg.td_lambda
        self._value_weight = algo_cfg.value_weight
        self._entropy_weight = algo_cfg.entropy_weight

        self._target_model = copy.deepcopy(self._model)
        self._target_model = model_wrap(
            self._target_model,
            wrapper_name='target',
            update_type='momentum',
            update_kwargs={'theta': algo_cfg.target_update_theta}
        )
        self._target_model = model_wrap(
            self._target_model,
            wrapper_name='hidden_state',
            state_num=self._cfg.learn.batch_size,
            init_fn=lambda: [None for _ in range(self._cfg.learn.agent_num)]
        )
        self._learn_model = model_wrap(
            self._model,
            wrapper_name='hidden_state',
            state_num=self._cfg.learn.batch_size,
            init_fn=lambda: [None for _ in range(self._cfg.learn.agent_num)]
        )
        self._learn_model.reset()
        self._target_model.reset()

    def _data_preprocess_learn(self, data: List[Any]) -> dict:
        r"""
        Overview:
            Preprocess the data to fit the required data format for learning

        Arguments:
            - data (:obj:`List[Dict[str, Any]]`): the data collected from collect function, the Dict
                in data should contain keys including at least ['obs', 'action', 'reward']

        Returns:
            - data (:obj:`Dict[str, Any]`): the processed data, including at least \
                ['obs', 'action', 'reward', 'done', 'weight']
        """
        # data preprocess
        data = timestep_collate(data)
        assert set(data.keys()) > set(['obs', 'action', 'reward'])
        if self._cuda:
            data = to_device(data, self._device)
        data['weight'] = data.get('weight', None)
        data['done'] = data['done'].float()
        return data

    def _forward_learn(self, data: dict) -> Dict[str, Any]:
        r"""
        Overview:
            Forward and backward function of learn mode, acquire the data and calculate the loss and\
            optimize learner model

        Arguments:
            - data (:obj:`dict`): Dict type data, including at least \
                ['obs', 'action', 'reward', 'done', 'weight']

        Returns:
            - info_dict (:obj:`Dict[str, Any]`): Including cur_lr and total_loss
                - cur_lr (:obj:`float`): Current learning rate
                - total_loss (:obj:`float`): The calculated loss
                - policy_loss (:obj:`float`): The policy(actor) loss of coma
                - value_loss (:obj:`float`): The value(critic) loss of coma
                - entropy_loss (:obj:`float`): The entropy loss
        """
        data = self._data_preprocess_learn(data)
        # forward
        self._learn_model.train()
        self._target_model.train()
        self._learn_model.reset(state=data['prev_state'][0])
        self._target_model.reset(state=data['prev_state'][0])
        q_value = self._learn_model.forward(data, mode='compute_critic')['q_value']
        with torch.no_grad():
            target_q_value = self._target_model.forward(data, mode='compute_critic')['q_value']
        logit = self._learn_model.forward(data, mode='compute_actor')['logit']

        data = coma_data(logit, data['action'], q_value, target_q_value, data['reward'], data['weight'])
        coma_loss = coma_error(data, self._gamma, self._lambda)
        total_loss = coma_loss.policy_loss + self._value_weight * coma_loss.q_value_loss - self._entropy_weight * \
            coma_loss.entropy_loss

        # update
        self._optimizer.zero_grad()
        total_loss.backward()
        self._optimizer.step()
        # after update
        self._target_model.update(self._learn_model.state_dict())
        return {
            'cur_lr': self._optimizer.defaults['lr'],
            'total_loss': total_loss.item(),
            'policy_loss': coma_loss.policy_loss.item(),
            'value_loss': coma_loss.q_value_loss.item(),
            'entropy_loss': coma_loss.entropy_loss.item(),
        }

    def _reset_learn(self, data_id: Optional[List[int]] = None) -> None:
        self._learn_model.reset(data_id=data_id)

    def _state_dict_learn(self) -> Dict[str, Any]:
        return {
            'model': self._learn_model.state_dict(),
            'optimizer': self._optimizer.state_dict(),
        }

    def _load_state_dict_learn(self, state_dict: Dict[str, Any]) -> None:
        self._learn_model.load_state_dict(state_dict['model'])
        self._optimizer.load_state_dict(state_dict['optimizer'])

    def _init_collect(self) -> None:
        r"""
        Overview:
            Collect mode init moethod. Called by ``self.__init__``.
            Init traj and unroll length, adder, collect model.
            Model has eps_greedy_sample wrapper and hidden state wrapper
        """
        self._unroll_len = self._cfg.collect.unroll_len
        self._adder = Adder(self._cuda, self._unroll_len)
        self._collect_model = model_wrap(
            self._model,
            wrapper_name='hidden_state',
            state_num=self._cfg.collect.env_num,
            save_prev_state=True,
            init_fn=lambda: [None for _ in range(self._cfg.learn.agent_num)]
        )
        self._collect_model = model_wrap(self._collect_model, wrapper_name='eps_greedy_sample')
        self._collect_model.reset()

    def _forward_collect(self, data: dict, eps: float) -> dict:
        r"""
        Overview:
            Collect output according to eps_greedy plugin

        Arguments:
            - data (:obj:`dict`): Dict type data, including at least ['obs'].

        Returns:
            - data (:obj:`dict`): The collected data
        """
        data_id = list(data.keys())
        data = default_collate(list(data.values()))
        if self._cuda:
            data = to_device(data, self._device)
        data = {'obs': data}
        self._collect_model.eval()
        with torch.no_grad():
            output = self._collect_model.forward(data, eps=eps, data_id=data_id, mode='compute_actor')
        if self._cuda:
            output = to_device(output, 'cpu')
        output = default_decollate(output)
        return {i: d for i, d in zip(data_id, output)}

    def _reset_collect(self, data_id: Optional[List[int]] = None) -> None:
        self._collect_model.reset(data_id=data_id)

    def _process_transition(self, obs: Any, model_output: dict, timestep: namedtuple) -> dict:
        r"""
        Overview:
            Generate dict type transition data from inputs.
        Arguments:
            - obs (:obj:`Any`): Env observation
            - model_output (:obj:`dict`): Output of collect model, including at least ['action', 'prev_state']
            - timestep (:obj:`namedtuple`): Output after env step, including at least ['obs', 'reward', 'done'] \
                (here 'obs' indicates obs after env step).
        Returns:
            - transition (:obj:`dict`): Dict type transition data.
        """
        transition = {
            'obs': obs,
            'next_obs': timestep.obs,
            'prev_state': model_output['prev_state'],
            'action': model_output['action'],
            'reward': timestep.reward,
            'done': timestep.done,
        }
        return EasyDict(transition)

    def _init_eval(self) -> None:
        r"""
        Overview:
            Evaluate mode init method. Called by ``self.__init__``.
            Init eval model with argmax strategy and hidden_state plugin.
        """
        self._eval_model = model_wrap(
            self._model,
            wrapper_name='hidden_state',
            state_num=self._cfg.eval.env_num,
            save_prev_state=True,
            init_fn=lambda: [None for _ in range(self._cfg.learn.agent_num)]
        )
        self._eval_model = model_wrap(self._eval_model, wrapper_name='argmax_sample')
        self._eval_model.reset()

    def _forward_eval(self, data: dict) -> dict:
        r"""
        Overview:
            Forward function of collect mode, similar to ``self._forward_collect``.

        Arguments:
            - data (:obj:`dict`): Dict type data, including at least ['obs'].

        Returns:
            - output (:obj:`dict`): Dict type data, including at least inferred action according to input obs.
        """
        data_id = list(data.keys())
        data = default_collate(list(data.values()))
        if self._cuda:
            data = to_device(data, self._device)
        data = {'obs': data}
        self._eval_model.eval()
        with torch.no_grad():
            output = self._eval_model.forward(data, data_id=data_id, mode='compute_actor')
        if self._cuda:
            output = to_device(output, 'cpu')
        output = default_decollate(output)
        return {i: d for i, d in zip(data_id, output)}

    def _reset_eval(self, data_id: Optional[List[int]] = None) -> None:
        self._eval_model.reset(data_id=data_id)

    def _get_train_sample(self, data: deque) -> Union[None, List[Any]]:
        r"""
        Overview:
            Get the trajectory, then sample from trajectory

        Arguments:
            - data (:obj:`deque`): The trajectory's cache

        Returns:
            - samples (:obj:`dict`): The training samples generated
        """
        return self._adder.get_train_sample(data)

    def default_model(self) -> Tuple[str, List[str]]:
        return 'coma', ['nervex.model.coma.coma']

    def _monitor_vars_learn(self) -> List[str]:
        return super()._monitor_vars_learn() + ['policy_loss', 'value_loss', 'entropy_loss']
