from typing import List, Dict, Any, Tuple
from collections import namedtuple
import copy
import torch

from ding.torch_utils import Adam, to_device
from ding.rl_utils import q_nstep_td_data, q_nstep_td_error, get_nstep_return_data, get_train_sample
from ding.model import model_wrap
from ding.utils import POLICY_REGISTRY
from ding.utils.data import default_collate, default_decollate
from .base_policy import Policy
from .common_utils import default_preprocess_learn


@POLICY_REGISTRY.register('pdqn')
class PDQNPolicy(Policy):
    r""":
    Overview:
        Policy class of PDQN algorithm, which extends the DQN algorithm on discrete-continuous hybrid action spaces.

    Config:
        == ==================== ======== ============== ======================================== =======================
        ID Symbol               Type     Default Value  Description                              Other(Shape)
        == ==================== ======== ============== ======================================== =======================
        1  ``type``             str      pdqn           | RL policy register name, refer to      | This arg is optional,
                                                        | registry ``POLICY_REGISTRY``           | a placeholder
        2  ``cuda``             bool     False          | Whether to use cuda for network        | This arg can be diff-
                                                                                                 | erent from modes
        3  ``on_policy``        bool     False          | Whether the RL algorithm is on-policy  | This value is always
                                                        | or off-policy                          | False for PDQN
        4  ``priority``         bool     False          | Whether use priority(PER)              | Priority sample,
                                                                                                 | update priority
        5  | ``priority_IS``    bool     False          | Whether use Importance Sampling Weight
           | ``_weight``                                | to correct biased update. If True,
                                                        | priority must be True.
        6  | ``discount_``      float    0.97,          | Reward's future discount factor, aka.  | May be 1 when sparse
           | ``factor``                  [0.95, 0.999]  | gamma                                  | reward env

        7  ``nstep``            int      1,             | N-step reward discount sum for target
                                         [3, 5]         | q_value estimation
        8  | ``learn.update``   int      3              | How many updates(iterations) to train  | This args can be vary
           | ``per_collect``                            | after collector's one collection. Only | from envs. Bigger val
                                                        | valid in serial training               | means more off-policy
        9  | ``learn.batch_``   int      64             | The number of samples of an iteration
           | ``size``
        10 | ``learn.multi``    bool     False          | whether to use multi gpu during
           | ``_gpu``
        11 | ``learn.learning`` float    0.001          | Gradient step length of an iteration.
           | ``_rate``
        12 | ``learn.target_``  int      100            | Frequence of target network update.    | Hard(assign) update
           | ``update_freq``
        13 | ``learn.ignore_``  bool     False          | Whether ignore done for target value   | Enable it for some
           | ``done``                                   | calculation.                           | fake termination env
        14 ``collect.n_sample`` int      [8, 128]       | The number of training samples of a    | It varies from
                                                        | call of collector.                     | different envs
        15 | ``collect.unroll`` int      1              | unroll length of an iteration          | In RNN, unroll_len>1
           | ``_len``
        16 | ``collect.noise``  float    0.1            | add noise to continuous args
           | ``_sigma``                                 | during collection
        17 | ``other.eps.type`` str      exp            | exploration rate decay type            | Support ['exp',
                                                                                                 | 'linear'].
        18 | ``other.eps.       float    0.95           | start value of exploration rate        | [0,1]
           |  start``
        19 | ``other.eps.       float    0.05           | end value of exploration rate          | [0,1]
           |  end``
        20 | ``other.eps.       int      10000          | decay length of exploration            | greater than 0. set
           |  decay``                                                                            | decay=10000 means
                                                                                                 | the exploration rate
                                                                                                 | decay from start
                                                                                                 | value to end value
                                                                                                 | during decay length.
        == ==================== ======== ============== ======================================== =======================
    """
    config = dict(
        type='pdqn',
        cuda=False,
        on_policy=False,
        priority=False,
        # (bool) Whether use Importance Sampling Weight to correct biased update. If True, priority must be True.
        priority_IS_weight=False,
        discount_factor=0.97,
        nstep=1,
        learn=dict(
            # (bool) Whether to use multi gpu
            multi_gpu=False,
            # How many updates(iterations) to train after collector's one collection.
            # Bigger "update_per_collect" means bigger off-policy.
            # collect data -> update policy-> collect data -> ...
            update_per_collect=3,
            batch_size=64,
            learning_rate=0.001,
            # ==============================================================
            # The following configs are algorithm-specific
            # ==============================================================
            # (int) Frequence of target network update.
            target_theta=0.005,
            # (bool) Whether ignore done(usually for max step termination env)
            ignore_done=False,
        ),
        # collect_mode config
        collect=dict(
            # (int) Only one of [n_sample, n_episode] shoule be set
            # n_sample=8,
            # (int) Cut trajectories into pieces with length "unroll_len".
            unroll_len=1,
            # (float) It is a must to add noise during collection. So here omits "noise" and only set "noise_sigma".
            noise_sigma=0.1,
        ),
        eval=dict(),
        # other config
        other=dict(
            # Epsilon greedy with decay.
            eps=dict(
                # (str) Decay type. Support ['exp', 'linear'].
                type='exp',
                start=0.95,
                end=0.1,
                # (int) Decay length(env step)
                decay=10000,
            ),
            replay_buffer=dict(replay_buffer_size=10000, ),
        ),
    )

    def _init_learn(self) -> None:
        """
        Overview:
            Learn mode init method. Called by ``self.__init__``, initialize the optimizer, algorithm arguments, main \
            and target model.
        """
        self._priority = self._cfg.priority
        self._priority_IS_weight = self._cfg.priority_IS_weight
        # Optimizer
        self._dis_optimizer = Adam(
            list(self._model.dis_head.parameters()) + list(self._model.cont_encoder.parameters()),
            # this is very important to put cont_encoder.parameters in here.
            lr=self._cfg.learn.learning_rate_dis
        )
        self._cont_optimizer = Adam(list(self._model.cont_head.parameters()), lr=self._cfg.learn.learning_rate_cont)

        self._gamma = self._cfg.discount_factor
        self._nstep = self._cfg.nstep

        # use model_wrapper for specialized demands of different modes
        self._target_model = copy.deepcopy(self._model)
        self._target_model = model_wrap(
            self._target_model,
            wrapper_name='target',
            update_type='momentum',
            update_kwargs={'theta': self._cfg.learn.target_theta}
        )
        self._learn_model = model_wrap(self._model, wrapper_name='hybrid_argmax_sample')
        self._learn_model.reset()
        self._target_model.reset()
        self.cont_train_cnt = 0
        self.disc_train_cnt = 0
        self.train_cnt = 0

    def _forward_learn(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Overview:
            Forward computation graph of learn mode(updating policy).
        Arguments:
            - data (:obj:`Dict[str, Any]`): Dict type data, a batch of data for training, values are torch.Tensor or \
                np.ndarray or dict/list combinations.
        Returns:
            - info_dict (:obj:`Dict[str, Any]`): Dict type data, a info dict indicated training result, which will be \
                recorded in text log and tensorboard, values are python scalar or a list of scalars.
        ArgumentsKeys:
            - necessary: ``obs``, ``action``, ``reward``, ``next_obs``, ``done``
            - optional: ``value_gamma``, ``IS``
        ReturnsKeys:
            - necessary: ``cur_lr``, ``q_loss``, ``continuous_loss``,
                         ``q_value``, ``priority``, ``reward``, ``target_q_value``
        """
        data = default_preprocess_learn(
            data,
            use_priority=self._priority,
            use_priority_IS_weight=self._cfg.priority_IS_weight,
            ignore_done=self._cfg.learn.ignore_done,
            use_nstep=True
        )
        if self._cuda:
            data = to_device(data, self._device)

        self.train_cnt += 1
        # ================================
        # Continuous args network forward
        # ================================
        if self.train_cnt == 1 or self.train_cnt % self._cfg.learn.update_circle in range(5, 10):
            dis_loss = torch.Tensor([0])
            td_error_per_sample = torch.Tensor([0])
            target_q_value = torch.Tensor([0])

            action_args = self._learn_model.forward(data['obs'], mode='compute_continuous')['action_args']

            # Current q value (main model) for cont loss
            discrete_inputs = {'state': data['obs'], 'action_args': action_args}
            # with torch.no_grad():
            q_pi_action_value = self._learn_model.forward(discrete_inputs, mode='compute_discrete')['logit']
            cont_loss = -q_pi_action_value.sum(dim=-1).mean()

            # ================================
            # Continuous args network update
            # ================================
            self._cont_optimizer.zero_grad()
            cont_loss.backward()
            self._cont_optimizer.step()

        # ====================
        # Q-learning forward
        # ====================
        if self.train_cnt == 1 or self.train_cnt % self._cfg.learn.update_circle in range(0, 5):
            cont_loss = torch.Tensor([0])
            q_pi_action_value = torch.Tensor([0])
            self._learn_model.train()
            self._target_model.train()
            # Current q value (main model)
            discrete_inputs = {'state': data['obs'], 'action_args': data['action']['action_args']}
            q_data_action_args_value = self._learn_model.forward(discrete_inputs, mode='compute_discrete')['logit']

            # Target q value
            with torch.no_grad():
                next_action_args = self._learn_model.forward(data['next_obs'], mode='compute_continuous')['action_args']
                next_action_args_cp = next_action_args.clone().detach()
                next_discrete_inputs = {'state': data['next_obs'], 'action_args': next_action_args_cp}
                target_q_value = self._target_model.forward(next_discrete_inputs, mode='compute_discrete')['logit']
                # Max q value action (main model)
                target_q_discrete_action = self._learn_model.forward(
                    next_discrete_inputs, mode='compute_discrete'
                )['action']['action_type']

            data_n = q_nstep_td_data(
                q_data_action_args_value, target_q_value, data['action']['action_type'], target_q_discrete_action,
                data['reward'], data['done'], data['weight']
            )
            value_gamma = data.get('value_gamma')
            dis_loss, td_error_per_sample = q_nstep_td_error(
                data_n, self._gamma, nstep=self._nstep, value_gamma=value_gamma
            )

            # ====================
            # Q-learning update
            # ====================
            self._dis_optimizer.zero_grad()
            dis_loss.backward()
            self._dis_optimizer.step()

            # =============
            # after update
            # =============
            self._target_model.update(self._learn_model.state_dict())

        return {
            'cur_lr': self._dis_optimizer.defaults['lr'],
            'q_loss': dis_loss.item(),
            'total_loss': (cont_loss + dis_loss).item(),
            'continuous_loss': cont_loss.item(),
            'q_value': q_pi_action_value.mean().item(),
            'priority': td_error_per_sample.abs().tolist(),
            'reward': data['reward'].mean().item(),
            'target_q_value': target_q_value.mean().item(),
        }

    def _state_dict_learn(self) -> Dict[str, Any]:
        """
        Overview:
            Return the state_dict of learn mode, usually including model and optimizer.
        Returns:
            - state_dict (:obj:`Dict[str, Any]`): the dict of current policy learn state, for saving and restoring.
        """
        return {
            'model': self._learn_model.state_dict(),
            'target_model': self._target_model.state_dict(),
            'dis_optimizer': self._dis_optimizer.state_dict(),
            'cont_optimizer': self._cont_optimizer.state_dict()
        }

    def _load_state_dict_learn(self, state_dict: Dict[str, Any]) -> None:
        """
        Overview:
            Load the state_dict variable into policy learn mode.
        Arguments:
            - state_dict (:obj:`Dict[str, Any]`): the dict of policy learn state saved before.

        .. tip::
            If you want to only load some parts of model, you can simply set the ``strict`` argument in \
            load_state_dict to ``False``, or refer to ``ding.torch_utils.checkpoint_helper`` for more \
            complicated operation.
        """
        self._learn_model.load_state_dict(state_dict['model'])
        self._target_model.load_state_dict(state_dict['target_model'])
        self._dis_optimizer.load_state_dict(state_dict['dis_optimizer'])
        self._cont_optimizer.load_state_dict(state_dict['cont_optimizer'])

    def _init_collect(self) -> None:
        """
        Overview:
            Collect mode init method. Called by ``self.__init__``, initialize algorithm arguments and collect_model, \
            enable the eps_greedy_sample for exploration.
        """
        self._unroll_len = self._cfg.collect.unroll_len
        self._gamma = self._cfg.discount_factor  # necessary for parallel
        self._nstep = self._cfg.nstep  # necessary for parallel
        self._collect_model = model_wrap(
            self._model,
            wrapper_name='action_noise',
            noise_type='gauss',
            noise_kwargs={
                'mu': 0.0,
                'sigma': self._cfg.collect.noise_sigma
            },
            noise_range=None
        )
        self._collect_model = model_wrap(self._collect_model, wrapper_name='hybrid_eps_greedy_multinomial_sample')
        self._collect_model.reset()

    def _forward_collect(self, data: Dict[int, Any], eps: float) -> Dict[int, Any]:
        """
        Overview:
            Forward computation graph of collect mode(collect training data), with eps_greedy for exploration.
        Arguments:
            - data (:obj:`Dict[str, Any]`): Dict type data, stacked env data for predicting policy_output(action), \
                values are torch.Tensor or np.ndarray or dict/list combinations, keys are env_id indicated by integer.
            - eps (:obj:`float`): epsilon value for exploration, which is decayed by collected env step.
        Returns:
            - output (:obj:`Dict[int, Any]`): The dict of predicting policy_output(action) for the interaction with \
                env and the constructing of transition.
        ArgumentsKeys:
            - necessary: ``obs``
        ReturnsKeys
            - necessary: ``logit``, ``action``
        """
        data_id = list(data.keys())
        data = default_collate(list(data.values()))
        if self._cuda:
            data = to_device(data, self._device)
        self._collect_model.eval()
        with torch.no_grad():
            action_args = self._collect_model.forward(data, 'compute_continuous', eps=eps)['action_args']
            inputs = {'state': data, 'action_args': action_args.clone().detach()}
            output = self._collect_model.forward(inputs, 'compute_discrete', eps=eps)
        if self._cuda:
            output = to_device(output, 'cpu')
        output = default_decollate(output)
        return {i: d for i, d in zip(data_id, output)}

    def _get_train_sample(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Overview:
            For a given trajectory(transitions, a list of transition) data, process it into a list of sample that \
            can be used for training directly. A train sample can be a processed transition(DQN with nstep TD) \
            or some continuous transitions(DRQN).
        Arguments:
            - data (:obj:`List[Dict[str, Any]`): The trajectory data(a list of transition), each element is the same \
                format as the return value of ``self._process_transition`` method.
        Returns:
            - samples (:obj:`dict`): The list of training samples.

        .. note::
            We will vectorize ``process_transition`` and ``get_train_sample`` method in the following release version. \
            And the user can customize the this data processing procecure by overriding this two methods and collector \
            itself.
        """
        data = get_nstep_return_data(data, self._nstep, gamma=self._gamma)
        return get_train_sample(data, self._unroll_len)

    def _process_transition(self, obs: Any, model_output: Dict[str, Any], timestep: namedtuple) -> Dict[str, Any]:
        """
        Overview:
            Generate a transition(e.g.: <s, a, s', r, d>) for this algorithm training.
        Arguments:
            - obs (:obj:`Any`): Env observation.
            - policy_output (:obj:`Dict[str, Any]`): The output of policy collect mode(``self._forward_collect``),\
                including at least ``action``.
            - timestep (:obj:`namedtuple`): The output after env step(execute policy output action), including at \
                least ``obs``, ``reward``, ``done``, (here obs indicates obs after env step).
        Returns:
            - transition (:obj:`dict`): Dict type transition data.
        """
        transition = {
            'obs': obs,
            'next_obs': timestep.obs,
            'action': model_output['action'],
            'logit': model_output['logit'],
            'reward': timestep.reward,
            'done': timestep.done,
        }
        return transition

    def _init_eval(self) -> None:
        r"""
        Overview:
            Evaluate mode init method. Called by ``self.__init__``, initialize eval_model.
        """
        self._eval_model = model_wrap(self._model, wrapper_name='hybrid_argmax_sample')
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
            - optional: ``logit``
        """
        data_id = list(data.keys())
        data = default_collate(list(data.values()))
        if self._cuda:
            data = to_device(data, self._device)
        self._eval_model.eval()
        with torch.no_grad():
            action_args = self._eval_model.forward(data, mode='compute_continuous')['action_args']
            inputs = {'state': data, 'action_args': action_args.clone().detach()}
            output = self._eval_model.forward(inputs, mode='compute_discrete')
        if self._cuda:
            output = to_device(output, 'cpu')
        output = default_decollate(output)
        return {i: d for i, d in zip(data_id, output)}

    def default_model(self) -> Tuple[str, List[str]]:
        """
        Overview:
            Return this algorithm default model setting for demonstration.
        Returns:
            - model_info (:obj:`Tuple[str, List[str]]`): model name and mode import_names

        .. note::
            The user can define and use customized network model but must obey the same inferface definition indicated \
            by import_names path. For PDQN, ``ding.model.template.pdqn.PDQN``
        """
        return 'pdqn', ['ding.model.template.pdqn']

    def _monitor_vars_learn(self) -> List[str]:
        r"""
        Overview:
            Return variables' names if variables are to used in monitor.
        Returns:
            - vars (:obj:`List[str]`): Variables' name list.
        """
        return ['cur_lr', 'total_loss', 'q_loss', 'continuous_loss', 'q_value', 'reward', 'target_q_value']
