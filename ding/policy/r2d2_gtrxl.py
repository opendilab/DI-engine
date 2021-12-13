import copy
from collections import namedtuple
from typing import List, Dict, Any, Tuple, Union, Optional

from ding.model import model_wrap
from ding.rl_utils import q_nstep_td_data, q_nstep_td_error, q_nstep_td_error_with_rescale, get_nstep_return_data, \
    get_train_sample
from ding.torch_utils import Adam, to_device
from ding.utils import POLICY_REGISTRY
from ding.utils.data import timestep_collate, default_collate, default_decollate
from .base_policy import Policy

from ding.model.common.head import *
from ding.torch_utils.network.gtrxl import GTrXL

from ding.utils import MODEL_REGISTRY


@MODEL_REGISTRY.register('gtrxl_discrete')
class GTrXLDiscreteHead(nn.Module):
    def __init__(
        self,
        obs_shape: int,
        action_shape: int,
        head_layer_num: int = 1,
        att_head_dim: int = 16,
        embedding_dim: int = 16,
        att_head_num: int = 2,
        att_mlp_num: int = 2,
        att_layer_num: int = 3,
        memory_len: int = 64,
        activation: Optional[nn.Module] = nn.ReLU(),
        head_norm_type: Optional[str] = None,
        head_noise: Optional[bool] = False,
        dropout: float = 0.,
    ) -> None:
        super(GTrXLDiscreteHead, self).__init__()
        self.core = GTrXL(
            input_dim=obs_shape,
            head_dim=att_head_dim,
            embedding_dim=embedding_dim,
            head_num=att_head_num,
            mlp_num=att_mlp_num,
            layer_num=att_layer_num,
            memory_len=memory_len,
            activation=activation,
            dropout_ratio=dropout,
        )
        self.head = DiscreteHead(
            hidden_size=embedding_dim,
            output_size=action_shape,
            layer_num=head_layer_num,
            activation=activation,
            norm_type=head_norm_type,
            noise=head_noise
        )

    def forward(self, x: torch.Tensor) -> Dict:
        o1 = self.core(x)
        o2 = self.head(o1['logit'])
        o2['memory'] = torch.permute(o1['memory'], (2, 0, 1, 3))  # bs first
        o2['transformer_out'] = o1['logit']
        return o2

    def reset(self, state: Optional[torch.Tensor] = None):
        print('reset')
        self.core.reset(state)
        print(self.core.memory)


@POLICY_REGISTRY.register('r2d2_gtrxl')
class R2D2GTrXLPolicy(Policy):
    r"""
    Overview:
        Policy class of R2D2, from paper `Recurrent Experience Replay in Distributed Reinforcement Learning` .
        R2D2 proposes that several tricks should be used to improve upon DRQN,
        namely some recurrent experience replay tricks such as burn-in.

    Config:
        == ==================== ======== ============== ======================================== =======================
        ID Symbol               Type     Default Value  Description                              Other(Shape)
        == ==================== ======== ============== ======================================== =======================
        1  ``type``             str      dqn            | RL policy register name, refer to      | This arg is optional,
                                                        | registry ``POLICY_REGISTRY``           | a placeholder
        2  ``cuda``             bool     False          | Whether to use cuda for network        | This arg can be diff-
                                                                                                 | erent from modes
        3  ``on_policy``        bool     False          | Whether the RL algorithm is on-policy
                                                        | or off-policy
        4  ``priority``         bool     False          | Whether use priority(PER)              | Priority sample,
                                                                                                 | update priority
        5  | ``priority_IS``    bool     False          | Whether use Importance Sampling Weight
           | ``_weight``                                | to correct biased update. If True,
                                                        | priority must be True.
        6  | ``discount_``      float    0.997,         | Reward's future discount factor, aka.  | May be 1 when sparse
           | ``factor``                  [0.95, 0.999]  | gamma                                  | reward env
        7  ``nstep``            int      3,             | N-step reward discount sum for target
                                         [3, 5]         | q_value estimation
        8  ``burnin_step``      int      2              | The timestep of burnin operation,
                                                        | which is designed to RNN hidden state
                                                        | difference caused by off-policy
        9  | ``learn.update``   int      1              | How many updates(iterations) to train  | This args can be vary
           | ``per_collect``                            | after collector's one collection. Only | from envs. Bigger val
                                                        | valid in serial training               | means more off-policy
        10 | ``learn.batch_``   int      64             | The number of samples of an iteration
           | ``size``
        11 | ``learn.learning`` float    0.001          | Gradient step length of an iteration.
           | ``_rate``
        12 | ``learn.value_``   bool     True           | Whether use value_rescale function for
           | ``rescale``                                | predicted value
        13 | ``learn.target_``  int      100            | Frequence of target network update.    | Hard(assign) update
           | ``update_freq``
        14 | ``learn.ignore_``  bool     False          | Whether ignore done for target value   | Enable it for some
           | ``done``                                   | calculation.                           | fake termination env
        15 ``collect.n_sample`` int      [8, 128]       | The number of training samples of a    | It varies from
                                                        | call of collector.                     | different envs
        16 | ``collect.unroll`` int      1              | unroll length of an iteration          | In RNN, unroll_len>1
           | ``_len``
        == ==================== ======== ============== ======================================== =======================
    """
    config = dict(
        # (str) RL policy register name (refer to function "POLICY_REGISTRY").
        type='r2d2',
        # (bool) Whether to use cuda for network.
        cuda=False,
        # (bool) Whether the RL algorithm is on-policy or off-policy.
        on_policy=False,
        # (bool) Whether use priority(priority sample, IS weight, update priority)
        priority=True,
        # (bool) Whether use Importance Sampling Weight to correct biased update. If True, priority must be True.
        priority_IS_weight=True,
        # ==============================================================
        # The following configs are algorithm-specific
        # ==============================================================
        # (float) Reward's future discount factor, aka. gamma.
        discount_factor=0.997,
        # (int) N-step reward for target q_value estimation
        nstep=5,
        # (int) the trajectory length to unroll the RNN network
        unroll_len=20,
        learn=dict(
            # (bool) Whether to use multi gpu
            multi_gpu=False,
            update_per_collect=1,
            batch_size=64,
            learning_rate=0.0001,
            # ==============================================================
            # The following configs are algorithm-specific
            # ==============================================================
            # (int) Frequence of target network update.
            # target_update_freq=100,
            target_update_theta=0.001,
            ignore_done=False,
        ),
        collect=dict(
            # NOTE it is important that don't include key n_sample here, to make sure self._traj_len=INF
            each_iter_n_sample=32,
            # `env_num` is used in hidden state, should equal to that one in env config.
            # User should specify this value in user config.
            env_num=None,
        ),
        eval=dict(
            # `env_num` is used in hidden state, should equal to that one in env config.
            # User should specify this value in user config.
            env_num=None,
        ),
        other=dict(
            eps=dict(
                type='exp',
                start=0.95,
                end=0.05,
                decay=10000,
            ),
            replay_buffer=dict(replay_buffer_size=10000, ),
        ),
    )

    def _init_learn(self) -> None:
        r"""
        Overview:
            Init the learner model of R2D2Policy

        Arguments:
            .. note::

                The _init_learn method takes the argument from the self._cfg.learn in the config file

            - learning_rate (:obj:`float`): The learning rate fo the optimizer
            - gamma (:obj:`float`): The discount factor
            - nstep (:obj:`int`): The num of n step return
            - value_rescale (:obj:`bool`): Whether to use value rescaled loss in algorithm
        """
        print('gtrxl')
        self._priority = self._cfg.priority
        self._priority_IS_weight = self._cfg.priority_IS_weight
        self._optimizer = Adam(self._model.parameters(), lr=self._cfg.learn.learning_rate)
        self._gamma = self._cfg.discount_factor
        self._nstep = self._cfg.nstep
        self._batch_size = self._cfg.learn.batch_size

        self._target_model = copy.deepcopy(self._model)
        # here we should not adopt the 'assign' mode of target network here because the reset bug
        # self._target_model = model_wrap(
        #     self._target_model,
        #     wrapper_name='target',
        #     update_type='assign',
        #     update_kwargs={'freq': self._cfg.learn.target_update_freq}
        # )
        self._target_model = model_wrap(
            self._target_model,
            wrapper_name='target',
            update_type='momentum',
            update_kwargs={'theta': self._cfg.learn.target_update_theta}
        )

        self._learn_model = model_wrap(self._model, wrapper_name='argmax_sample')
        self._learn_model.reset()
        self._target_model.reset()

    def _data_preprocess_learn(self, data: List[Dict[str, Any]]) -> dict:
        r"""
        Overview:
            Preprocess the data to fit the required data format for learning

        Arguments:
            - data (:obj:`List[Dict[str, Any]]`): the data collected from collect function

        Returns:
            - data (:obj:`Dict[str, Any]`): the processed data, including at least \
                ['main_obs', 'target_obs', 'action', 'reward', 'done', 'weight']
            - data_info (:obj:`dict`): the data info, such as replay_buffer_idx, replay_unique_id
        """
        # data preprocess
        from copy import deepcopy
        prev_mem = [b.pop('prev_memory')[0] for b in deepcopy(data)]  # retrieve the memory corresponding to the first observation in each trajectory
        prev_mem = torch.stack(prev_mem, 0).permute(1, 2, 0, 3)  # (layer_num, memory_len, bs, embedding_dim)
        data = timestep_collate(data)
        data['prev_memory'] = prev_mem
        if self._cuda:
            data = to_device(data, self._device)

        if self._priority_IS_weight:
            assert self._priority, "Use IS Weight correction, but Priority is not used."
        if self._priority and self._priority_IS_weight:
            data['weight'] = data['IS']
        else:
            data['weight'] = data.get('weight', None)

        # data['done'], data['weight'], data['value_gamma'] is used in def _forward_learn() to calculate
        # the q_nstep_td_error, should be length of [self._unroll_len]
        ignore_done = self._cfg.learn.ignore_done
        if ignore_done:
            data['done'] = [None for _ in range(self._unroll_len)]
        else:
            data['done'] = data['done'].float()  # for computation of online model self._learn_model
            # NOTE that after the proprocessing of  get_nstep_return_data() in _get_train_sample
            # the data['done'][t] is already the n-step done

        # if the data don't include 'weight' or 'value_gamma' then fill in None in a list
        # with length of [self._unroll_len_add_burnin_step-self._burnin_step],
        # below is two different implementation ways
        if 'value_gamma' not in data:
            data['value_gamma'] = [None for _ in range(self._unroll_len)]
        else:
            data['value_gamma'] = data['value_gamma']

        if 'weight' not in data or data['weight'] is None:
            data['weight'] = [None for _ in range(self._unroll_len)]
        else:
            data['weight'] = data['weight'] * torch.ones_like(data['done'])
            # every timestep in sequence has same weight, which is the _priority_IS_weight in PER

        data['action'] = data['action'][:-self._nstep]
        data['reward'] = data['reward'][:-self._nstep]

        data['main_obs'] = data['obs'][:-self._nstep]
        # the target_obs is used to calculate the target_q_value
        data['target_obs'] = data['obs'][self._nstep:]

        return data

    def _forward_learn(self, data: dict) -> Dict[str, Any]:
        r"""
        Overview:
            Forward and backward function of learn mode.
            Acquire the data, calculate the loss and optimize learner model.

        Arguments:
            - data (:obj:`dict`): Dict type data, including at least \
                ['main_obs', 'target_obs', 'action', 'reward', 'done', 'weight']

        Returns:
            - info_dict (:obj:`Dict[str, Any]`): Including cur_lr and total_loss
                - cur_lr (:obj:`float`): Current learning rate
                - total_loss (:obj:`float`): The calculated loss
        """
        # forward
        data = self._data_preprocess_learn(data)  # shape (seq_len, bs, obs_dim)
        self._learn_model.train()
        self._target_model.train()
        # use the previous hidden state memory
        self._learn_model._model.core.reset(len(data), data['prev_memory'])
        self._target_model._model.core.reset(len(data), data['prev_memory'])

        inputs = data['main_obs']
        q_value = self._learn_model.forward(inputs)['logit']  # shape (seq_len, bs, act_dim)

        next_inputs = data['target_obs']
        with torch.no_grad():
            target_q_value = self._target_model.forward(next_inputs)['logit']
            # argmax_action double_dqn
            target_q_action = self._learn_model.forward(next_inputs)['action']

        action, reward, done, weight = data['action'], data['reward'], data['done'], data['weight']
        value_gamma = data['value_gamma']
        # T, B, nstep -> T, nstep, B
        reward = reward.permute(0, 2, 1).contiguous()
        loss = []
        td_error = []
        for t in range(self._unroll_len - self._nstep):
            # here t=0 means timestep <self._burnin_step> in the original sample sequence, we minus self._nstep
            # because for the last <self._nstep> timestep in the sequence, we don't have their target obs
            td_data = q_nstep_td_data(
                q_value[t], target_q_value[t], action[t], target_q_action[t], reward[t], done[t], weight[t]
            )
            l, e = q_nstep_td_error(td_data, self._gamma, self._nstep, value_gamma=value_gamma[t])
            loss.append(l)
            td_error.append(e.abs())
        loss = sum(loss) / (len(loss) + 1e-8)

        # using the mixture of max and mean absolute n-step TD-errors as the priority of the sequence
        td_error_per_sample = 0.9 * torch.max(
            torch.stack(td_error), dim=0
        )[0] + (1 - 0.9) * (torch.sum(torch.stack(td_error), dim=0) / (len(td_error) + 1e-8))
        # td_error shape list(<self._unroll_len_add_burnin_step-self._burnin_step-self._nstep>, B), for example, (75,64)
        # torch.sum(torch.stack(td_error), dim=0) can also be replaced with sum(td_error)

        # update
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        # after update
        self._target_model.update(self._learn_model.state_dict())

        # the information for debug
        batch_range = torch.arange(action[0].shape[0])
        q_s_a_t0 = q_value[0][batch_range, action[0]]
        target_q_s_a_t0 = target_q_value[0][batch_range, target_q_action[0]]

        return {
            'cur_lr': self._optimizer.defaults['lr'],
            'total_loss': loss.item(),
            'priority': td_error_per_sample.abs().tolist(),
            # the first timestep in the sequence, may not be the start of episode
            'q_s_taken-a_t0': q_s_a_t0.mean().item(),
            'target_q_s_max-a_t0': target_q_s_a_t0.mean().item(),
            'q_s_a-mean_t0': q_value[0].mean().item(),
        }

    def _reset_learn(self, data_id: Optional[List[int]] = None) -> None:
        self._learn_model.reset()

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
            Collect mode init method. Called by ``self.__init__``.
            Init traj and unroll length, collect model.
        """
        assert 'unroll_len' not in self._cfg.collect, "Use default unroll_len"
        self._nstep = self._cfg.nstep
        self._gamma = self._cfg.discount_factor
        self._unroll_len = self._cfg.unroll_len
        self._collect_model = model_wrap(self._model, wrapper_name='transformer', seq_len=self._unroll_len)
        self._collect_model = model_wrap(self._collect_model, wrapper_name='eps_greedy_sample')
        self._collect_model.reset()

    def _forward_collect(self, data: dict, eps: float) -> dict:
        r"""
        Overview:
            Forward function for collect mode with eps_greedy
        Arguments:
            - data (:obj:`Dict[str, Any]`): Dict type data, stacked env data for predicting policy_output(action), \
                values are torch.Tensor or np.ndarray or dict/list combinations, keys are env_id indicated by integer.
            - eps (:obj:`float`): epsilon value for exploration, which is decayed by collected env step.
        Returns:
            - output (:obj:`Dict[int, Any]`): Dict type data, including at least inferred action according to input obs.
        ReturnsKeys
            - necessary: ``action``
        """
        data_id = list(data.keys())
        data = default_collate(list(data.values()))
        if self._cuda:
            data = to_device(data, self._device)
        self._collect_model.eval()
        with torch.no_grad():
            output = self._collect_model.forward(data, eps=eps)
        if self._cuda:
            output = to_device(output, 'cpu')
        output = default_decollate(output)
        return {i: d for i, d in zip(data_id, output)}

    def _reset_collect(self, data_id: Optional[List[int]] = None) -> None:
        self._collect_model.reset()

    def _process_transition(self, obs: Any, model_output: dict, timestep: namedtuple) -> dict:
        r"""
        Overview:
            Generate dict type transition data from inputs.
        Arguments:
            - obs (:obj:`Any`): Env observation
            - model_output (:obj:`dict`): Output of collect model, including at least ['action', 'prev_state']
            - timestep (:obj:`namedtuple`): Output after env step, including at least ['reward', 'done'] \
                (here 'obs' indicates obs after env step).
        Returns:
            - transition (:obj:`dict`): Dict type transition data.
        """
        transition = {
            'obs': obs,
            'action': model_output['action'],
            'prev_memory': model_output['memory'],
            'prev_state': None,
            'reward': timestep.reward,
            'done': timestep.done,
        }
        return transition

    def _get_train_sample(self, data: list) -> Union[None, List[Any]]:
        r"""
        Overview:
            Get the trajectory and the n step return data, then sample from the n_step return data

        Arguments:
            - data (:obj:`list`): The trajectory's cache

        Returns:
            - samples (:obj:`dict`): The training samples generated
        """
        data = get_nstep_return_data(data, self._nstep, gamma=self._gamma)
        return get_train_sample(data, self._unroll_len)

    def _init_eval(self) -> None:
        r"""
        Overview:
            Evaluate mode init method. Called by ``self.__init__``.
            Init eval model with argmax strategy.
        """
        self._eval_model = model_wrap(self._model, wrapper_name='transformer', seq_len=self._unroll_len)
        self._eval_model = model_wrap(self._eval_model, wrapper_name='argmax_sample')
        self._eval_model.reset()

    def _forward_eval(self, data: dict) -> dict:
        r"""
        Overview:
            Forward function of eval mode, similar to ``self._forward_collect``.
        Arguments:
            - data (:obj:`Dict[str, Any]`): Dict type data, stacked env data for predicting policy_output(action), \
                values are torch.Tensor or np.ndarray or dict/list combinations, keys are env_id indicated by integer.
        Returns:
            - output (:obj:`Dict[int, Any]`): The dict of predicting action for the interaction with env.
        ReturnsKeys
            - necessary: ``action``
        """
        data_id = list(data.keys())
        data = default_collate(list(data.values()))
        if self._cuda:
            data = to_device(data, self._device)
        self._eval_model.eval()
        with torch.no_grad():
            output = self._eval_model.forward(data)
        if self._cuda:
            output = to_device(output, 'cpu')
        output = default_decollate(output)
        return {i: d for i, d in zip(data_id, output)}

    def _reset_eval(self, data_id: Optional[List[int]] = None) -> None:
        self._eval_model.reset()

    def default_model(self) -> Tuple[str, List[str]]:
        return 'gtrxl_discrete', ['ding.policy.r2d2_gtrxl']

    def _monitor_vars_learn(self) -> List[str]:
        return super()._monitor_vars_learn() + [
            'total_loss', 'priority', 'q_s_taken-a_t0', 'target_q_s_max-a_t0', 'q_s_a-mean_t0'
        ]
