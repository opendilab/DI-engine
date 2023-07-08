from typing import List, Dict, Any, Tuple
from collections import namedtuple
import copy
import torch

from ding.torch_utils import Adam, RMSprop, to_device, ContrastiveLoss
from ding.rl_utils import q_nstep_td_data, q_nstep_td_error, get_nstep_return_data, get_train_sample
from ding.model import model_wrap
from ding.utils import POLICY_REGISTRY
from ding.utils.data import default_collate, default_decollate

from .dqn import DQNPolicy
from .common_utils import default_preprocess_learn


@POLICY_REGISTRY.register('averaged_dqn')
class AveragedDQNPolicy(DQNPolicy):
    """
    Overview:
        Policy class of Averaged_DQN algorithm.
        paper: https://arxiv.org/pdf/1611.01929.pdf

    Config:
        == ===================== ======== ============== ======================================= =======================
        ID Symbol                Type     Default Value  Description                              Other(Shape)
        == ===================== ======== ============== ======================================= =======================
        1  ``type``              str      averaged_dqn   | RL policy register name, refer to     | This arg is optional,
                                                         | registry ``POLICY_REGISTRY``          | a placeholder
        2  ``cuda``              bool     False          | Whether to use cuda for network       | This arg can be diff-
                                                                                                 | erent from modes
        3  ``on_policy``         bool     False          | Whether the RL algorithm is on-policy
                                                         | or off-policy
        4  ``priority``          bool     False          | Whether use priority(PER)             | Priority sample,
                                                                                                 | update priority
        5  | ``priority_IS``     bool     False          | Whether use Importance Sampling
           | ``_weight``                                 | Weight to correct biased update. If
                                                         | True, priority must be True.
        6  | ``discount_``       float    0.97,          | Reward's future discount factor, aka. | May be 1 when sparse
           | ``factor``                   [0.95, 0.999]  | gamma                                 | reward env
        7  ``nstep``             int      1,             | N-step reward discount sum for target
                                          [3, 5]         | q_value estimation
        8  | ``model.dueling``   bool     True           | dueling head architecture
        9  | ``model.encoder``   list     [32, 64,       | Sequence of ``hidden_size`` of        | default kernel_size
           | ``_hidden``         (int)    64, 128]       | subsequent conv layers and the        | is [8, 4, 3]
           | ``_size_list``                               | final dense layer.                   | default stride is
                                                                                                 | [4, 2 ,1]
        10 | ``learn.update``    int      3              | How many updates(iterations) to train | This args can be vary
           | ``per_collect``                             | after collector's one collection.     | from envs. Bigger val
                                                         | Only valid in serial training         | means more off-policy
        11 | ``learn.batch_``    int      64             | The number of samples of an iteration
           | ``size``
        12 | ``learn.learning``  float    0.001          | Gradient step length of an iteration.
           | ``_rate``
        13 | ``learn.target_``   int      100            | Frequence of target network update.   | Hard(assign) update
           | ``update_freq``
        14 | ``learn.target_``   float    0.005          | Frequence of target network update.   | Soft(assign) update
           | ``theta``                                   | Only one of [target_update_freq,
           |                                             | target_theta] should be set
        15 | ``learn.ignore_``   bool     False          | Whether ignore done for target value  | Enable it for some
           | ``done``                                    | calculation.                          | fake termination env
        16 ``collect.n_sample``  int      [8, 128]       | The number of training samples of a   | It varies from
                                                         | call of collector.                    | different envs
        17 ``collect.n_episode`` int      8              | The number of training episodes of a  | only one of [n_sample
                                                         | call of collector                     | ,n_episode] should
                                                         |                                       | be set
        18 | ``collect.unroll``  int      1              | unroll length of an iteration         | In RNN, unroll_len>1
           | ``_len``
        19 | ``other.eps.type``  str      exp            | exploration rate decay type           | Support ['exp',
                                                                                                 | 'linear'].
        20 | ``other.eps.``      float    0.95           | start value of exploration rate       | [0,1]
           | ``start``
        21 | ``other.eps.``      float    0.1            | end value of exploration rate         | [0,1]
           | ``end``
        22 | ``other.eps.``      int      10000          | decay length of exploration           | greater than 0. set
           | ``decay``                                                                           | decay=10000 means
                                                                                                 | the exploration rate
                                                                                                 | decay from start
                                                                                                 | value to end value
                                                                                                 | during decay length.
        == ===================== ======== ============== ======================================= =======================
    """

    config = dict(
        # (str) RL policy register name (refer to function "POLICY_REGISTRY").
        type='averaged_dqn',
        # (bool) Whether use cuda in policy.
        cuda=False,
        # (bool) Whether learning policy is the same as collecting data policy(on-policy).
        on_policy=False,
        # (bool) Whether enable priority experience sample.
        priority=False,
        # (bool) Whether use Importance Sampling Weight to correct biased update. If True, priority must be True.
        priority_IS_weight=False,
        # (float) Discount factor(gamma) for returns.
        discount_factor=0.97,
        # (int) The number of step for calculating target q_value.
        nstep=1,
        # (int) The number of previously Q-values used in current action-value estimate.
        num_of_prime=5,
        model=dict(
            #(list(int)) Sequence of ``hidden_size`` of subsequent conv layers and the final dense layer.
            encoder_hidden_size_list=[128, 128, 64],
        ),
        learn=dict(
            # (int) How many updates(iterations) to train after collector's one collection.
            # Bigger "update_per_collect" means bigger off-policy.
            # collect data -> update policy-> collect data -> ...
            update_per_collect=3,
            # (int) How many samples in a training batch.
            batch_size=64,
            # (float) The step size of gradient descent.
            learning_rate=0.001,
            # (int) Frequence of target network update.
            # Only one of [target_update_freq, target_theta] should be set.
            target_update_freq=100,
            # (float) : Used for soft update of the target network.
            # aka. Interpolation factor in EMA update for target network.
            # Only one of [target_update_freq, target_theta] should be set.
            target_theta=0.005,
            # (bool) Whether ignore done(usually for max step termination env).
            # Note: Gym wraps the MuJoCo envs by default with TimeLimit environment wrappers.
            # These limit HalfCheetah, and several other MuJoCo envs, to max length of 1000.
            # However, interaction with HalfCheetah always gets done with done is False,
            # Since we inplace done==True with done==False to keep
            # TD-error accurate computation(``gamma * (1 - done) * next_v + reward``),
            # when the episode step is greater than max episode step.
            ignore_done=False,
        ),
        # collect_mode config
        collect=dict(
            # (int) How many training samples collected in one collection procedure.
            # Only one of [n_sample, n_episode] shoule be set.
            n_sample=8,
            # (int) Split episodes or trajectories into pieces with length `unroll_len`.
            unroll_len=1,
        ),
        eval=dict(),
        # other config
        other=dict(
            # Epsilon greedy with decay.
            eps=dict(
                # (str) Decay type. Support ['exp', 'linear'].
                type='exp',
                # (float) Epsilon start value.
                start=0.95,
                # (float) Epsilon end value.
                end=0.1,
                # (int) Decay length(env step).
                decay=10000,
            ),
            replay_buffer=dict(
                # (int) Maximum size of replay buffer. Usually, larger buffer size is good.
                replay_buffer_size=10000,
            ),
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
        self._optimizer = Adam(self._model.parameters(), lr=self._cfg.learn.learning_rate)

        self._gamma = self._cfg.discount_factor
        self._nstep = self._cfg.nstep
        self._num_of_prime = self._cfg.num_of_prime
        
        # build the prime_model_list for averaged_DQN
        if not hasattr(self, '_prime_model_list'):
            self._prime_model_list = [copy.deepcopy(self._model) for _ in range(self._num_of_prime)]

        # use model_wrapper for specialized demands of different modes
        self._target_model_list = copy.deepcopy(self._prime_model_list)
        if 'target_update_freq' in self._cfg.learn: 
            for idx, target_model in enumerate(self._target_model_list):
                self._target_model_list[idx] = model_wrap(
                    target_model,
                    wrapper_name='target',
                    update_type='assign',
                    update_kwargs={'freq': self._cfg.learn.target_update_freq}
                )
        elif 'target_theta' in self._cfg.learn:
            for idx, target_model in enumerate(self._target_model_list):
                self._target_model_list[idx] = model_wrap(
                    target_model,
                    wrapper_name='target',
                    update_type='momentum',
                    update_kwargs={'theta': self._cfg.learn.target_theta}
                )
        else:
            raise RuntimeError("DQN needs target network, please either indicate target_update_freq or target_theta")
        self._learn_model = model_wrap(self._model, wrapper_name='argmax_sample')
        self._learn_model.reset()
        for target_model in self._target_model_list:
            target_model.reset()

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
            - optional: ``value_gamma``
        ReturnsKeys:
            - necessary: ``cur_lr``, ``total_loss``, ``priority``
            - optional: ``action_distribution``
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
        # ====================
        # Q-learning forward
        # ====================
        self._learn_model.train()
        for prime_model in self._prime_model_list:
            prime_model.train()
        for target_model in self._target_model_list:
            target_model.train()
        # Current q value (main model)
        q_value = self._learn_model.forward(data['obs'])['logit']
        with torch.no_grad():
            # q values of prime models, skip [0] the same as learn_model
            for prime_model in self._prime_model_list[1:]:
                q_value += prime_model.forward(data['obs'])['logit']
        q_value /= self._num_of_prime

        # target q value
        with torch.no_grad():
            # target q value of target net (k prime)
            target_q_value = 0
            for target_model in self._target_model_list:
                target_q_value += target_model.forward(data['next_obs'])['logit']
            target_q_value /= self._num_of_prime
            # Max q value action (main model), i.e. Double DQN
            next_q_value = self._learn_model.forward(data['next_obs'])['logit']
            for prime_model in self._prime_model_list[1:]:
                next_q_value += prime_model.forward(data['next_obs'])['logit']
            next_q_value /= self._num_of_prime
            target_q_action = [v.argmax(dim=-1) for v in next_q_value]
            target_q_action = torch.stack(target_q_action).squeeze()

        data_n = q_nstep_td_data(
            q_value, target_q_value, data['action'], target_q_action, data['reward'], data['done'], data['weight']
        )
        value_gamma = data.get('value_gamma')
        loss, td_error_per_sample = q_nstep_td_error(data_n, self._gamma, nstep=self._nstep, value_gamma=value_gamma)

        # ====================
        # Q-learning update
        # ====================
        self._optimizer.zero_grad()
        loss.backward()
        if self._cfg.multi_gpu:
            self.sync_gradients(self._learn_model)
        self._optimizer.step()
        
        # update prime models
        for i in reversed(range(1, self._num_of_prime)):
            self._prime_model_list[i].load_state_dict(self._prime_model_list[i - 1].state_dict(), strict=True)
        self._prime_model_list[0].load_state_dict(self._learn_model.state_dict(), strict=True)
        
        # =============
        # after update
        # =============
        for idx in range(self._num_of_prime):
            self._target_model_list[idx].update(self._prime_model_list[idx].state_dict())
        
        return {
            'cur_lr': self._optimizer.defaults['lr'],
            'total_loss': loss.item(),
            'q_value': q_value.mean().item(),
            'target_q_value': target_q_value.mean().item(),
            'priority': td_error_per_sample.abs().tolist(),
            # Only discrete action satisfying len(data['action'])==1 can return this and draw histogram on tensorboard.
            # '[histogram]action_distribution': data['action'],
        }

    def _init_eval(self) -> None:
        r"""
        Overview:
            Evaluate mode init method. Called by ``self.__init__``, initialize eval_model.
        """
        self._num_of_prime = self._cfg.num_of_prime
        if not hasattr(self, '_prime_model_list'):
            self._prime_model_list = [copy.deepcopy(self._model) for _ in range(self._num_of_prime)]
        self._eval_model_list = self._prime_model_list

    def _forward_eval(self, data: Dict[int, Any]) -> Dict[int, Any]:
        """
        Overview:
            Forward computation graph of eval mode(evaluate policy performance), at most cases, it is similar to \
            ``self._forward_collect``.
        Arguments:
            - data (:obj:`Dict[str, Any]`): Dict type data, stacked env data for predicting policy_output(action), \
                values are torch.Tensor or np.ndarray or dict/list combinations, keys are env_id indicated by integer.
        Returns:
            - output (:obj:`Dict[int, Any]`): The dict of predicting action for the interaction with env.
        ArgumentsKeys:
            - necessary: ``obs``
        ReturnsKeys
            - necessary: ``action``
        """
        data_id = list(data.keys())
        data = default_collate(list(data.values()))
        if self._cuda:
            data = to_device(data, self._device)
        for eval_model in self._eval_model_list:
            eval_model.eval()
        with torch.no_grad():
            output = self._eval_model_list[0].forward(data)
            q_value = 0
            for eval_model in self._eval_model_list:
                q_value += eval_model.forward(data)['logit']
            q_value /= self._num_of_prime
            logit = q_value
            assert isinstance(logit, torch.Tensor) or isinstance(logit, list)
            if isinstance(logit, torch.Tensor):
                logit = [logit]
            output['logit'] = logit
            
            if 'action_mask' in output:
                mask = output['action_mask']
                if isinstance(mask, torch.Tensor):
                    mask = [mask]
                logit = [l.sub_(1e8 * (1 - m)) for l, m in zip(logit, mask)]
            action = [l.argmax(dim=-1) for l in logit]
            if len(action) == 1:
                action, logit = action[0], logit[0]
            output['action'] = action
        
        if self._cuda:
            output = to_device(output, 'cpu')
        output = default_decollate(output)
        return {i: d for i, d in zip(data_id, output)}


    def _state_dict_learn(self) -> Dict[str, Any]:
            """
            Overview:
                Return the state_dict of learn mode, usually including model and optimizer.
            Returns:
                - state_dict (:obj:`Dict[str, Any]`): the dict of current policy learn state, for saving and restoring.
            """
            prime_list = [learn_model.state_dict() for learn_model in self._prime_model_list]
            target_list = [learn_model.state_dict() for learn_model in self._target_model_list]
            return {
                'model': self._learn_model.state_dict(),
                'prime_list': prime_list,
                'target_list': target_list,
                'optimizer': self._optimizer.state_dict(),
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
        for idx in range(self._num_of_prime):
            self._learn_model_list[idx].load_state_dict(state_dict['prime_list'][idx])
            self._target_model_list[idx].load_state_dict(state_dict['target_list'][idx])
        self._learn_model.load_state_dict(state_dict['model'])
        self._optimizer.load_state_dict(state_dict['optimizer'])


@POLICY_REGISTRY.register('ensemble_dqn')
class EnsembleDQNPolicy(DQNPolicy):
    """
    Overview:
        Policy class of Ensemble_DQN algorithm.
        paper: https://arxiv.org/pdf/1611.01929.pdf

    Config:
        == ===================== ======== ============== ======================================= =======================
        ID Symbol                Type     Default Value  Description                              Other(Shape)
        == ===================== ======== ============== ======================================= =======================
        1  ``type``              str      ensemble_dqn   | RL policy register name, refer to     | This arg is optional,
                                                         | registry ``POLICY_REGISTRY``          | a placeholder
        2  ``cuda``              bool     False          | Whether to use cuda for network       | This arg can be diff-
                                                                                                 | erent from modes
        3  ``on_policy``         bool     False          | Whether the RL algorithm is on-policy
                                                         | or off-policy
        4  ``priority``          bool     False          | Whether use priority(PER)             | Priority sample,
                                                                                                 | update priority
        5  | ``priority_IS``     bool     False          | Whether use Importance Sampling
           | ``_weight``                                 | Weight to correct biased update. If
                                                         | True, priority must be True.
        6  | ``discount_``       float    0.97,          | Reward's future discount factor, aka. | May be 1 when sparse
           | ``factor``                   [0.95, 0.999]  | gamma                                 | reward env
        7  ``nstep``             int      1,             | N-step reward discount sum for target
                                          [3, 5]         | q_value estimation
        8  | ``model.dueling``   bool     True           | dueling head architecture
        9  | ``model.encoder``   list     [32, 64,       | Sequence of ``hidden_size`` of        | default kernel_size
           | ``_hidden``         (int)    64, 128]       | subsequent conv layers and the        | is [8, 4, 3]
           | ``_size_list``                               | final dense layer.                   | default stride is
                                                                                                 | [4, 2 ,1]
        10 | ``learn.update``    int      3              | How many updates(iterations) to train | This args can be vary
           | ``per_collect``                             | after collector's one collection.     | from envs. Bigger val
                                                         | Only valid in serial training         | means more off-policy
        11 | ``learn.batch_``    int      64             | The number of samples of an iteration
           | ``size``
        12 | ``learn.learning``  float    0.001          | Gradient step length of an iteration.
           | ``_rate``
        13 | ``learn.target_``   int      100            | Frequence of target network update.   | Hard(assign) update
           | ``update_freq``
        14 | ``learn.target_``   float    0.005          | Frequence of target network update.   | Soft(assign) update
           | ``theta``                                   | Only one of [target_update_freq,
           |                                             | target_theta] should be set
        15 | ``learn.ignore_``   bool     False          | Whether ignore done for target value  | Enable it for some
           | ``done``                                    | calculation.                          | fake termination env
        16 ``collect.n_sample``  int      [8, 128]       | The number of training samples of a   | It varies from
                                                         | call of collector.                    | different envs
        17 ``collect.n_episode`` int      8              | The number of training episodes of a  | only one of [n_sample
                                                         | call of collector                     | ,n_episode] should
                                                         |                                       | be set
        18 | ``collect.unroll``  int      1              | unroll length of an iteration         | In RNN, unroll_len>1
           | ``_len``
        19 | ``other.eps.type``  str      exp            | exploration rate decay type           | Support ['exp',
                                                                                                 | 'linear'].
        20 | ``other.eps.``      float    0.95           | start value of exploration rate       | [0,1]
           | ``start``
        21 | ``other.eps.``      float    0.1            | end value of exploration rate         | [0,1]
           | ``end``
        22 | ``other.eps.``      int      10000          | decay length of exploration           | greater than 0. set
           | ``decay``                                                                           | decay=10000 means
                                                                                                 | the exploration rate
                                                                                                 | decay from start
                                                                                                 | value to end value
                                                                                                 | during decay length.
        == ===================== ======== ============== ======================================= =======================
    """

    config = dict(
        # (str) RL policy register name (refer to function "POLICY_REGISTRY").
        type='ensemble_dqn',
        # (bool) Whether use cuda in policy.
        cuda=False,
        # (bool) Whether learning policy is the same as collecting data policy(on-policy).
        on_policy=False,
        # (bool) Whether enable priority experience sample.
        priority=False,
        # (bool) Whether use Importance Sampling Weight to correct biased update. If True, priority must be True.
        priority_IS_weight=False,
        # (float) Discount factor(gamma) for returns.
        discount_factor=0.97,
        # (int) The number of step for calculating target q_value.
        nstep=1,
        # (int) The number of training model
        num_of_model=5,
        model=dict(
            #(list(int)) Sequence of ``hidden_size`` of subsequent conv layers and the final dense layer.
            encoder_hidden_size_list=[128, 128, 64],
        ),
        learn=dict(
            # (int) How many updates(iterations) to train after collector's one collection.
            # Bigger "update_per_collect" means bigger off-policy.
            # collect data -> update policy-> collect data -> ...
            update_per_collect=3,
            # (int) How many samples in a training batch.
            batch_size=64,
            # (float) The step size of gradient descent.
            learning_rate=0.001,
            # (int) Frequence of target network update.
            # Only one of [target_update_freq, target_theta] should be set.
            target_update_freq=100,
            # (float) : Used for soft update of the target network.
            # aka. Interpolation factor in EMA update for target network.
            # Only one of [target_update_freq, target_theta] should be set.
            target_theta=0.005,
            # (bool) Whether ignore done(usually for max step termination env).
            # Note: Gym wraps the MuJoCo envs by default with TimeLimit environment wrappers.
            # These limit HalfCheetah, and several other MuJoCo envs, to max length of 1000.
            # However, interaction with HalfCheetah always gets done with done is False,
            # Since we inplace done==True with done==False to keep
            # TD-error accurate computation(``gamma * (1 - done) * next_v + reward``),
            # when the episode step is greater than max episode step.
            ignore_done=False,
        ),
        # collect_mode config
        collect=dict(
            # (int) How many training samples collected in one collection procedure.
            # Only one of [n_sample, n_episode] shoule be set.
            n_sample=8,
            # (int) Split episodes or trajectories into pieces with length `unroll_len`.
            unroll_len=1,
        ),
        eval=dict(),
        # other config
        other=dict(
            # Epsilon greedy with decay.
            eps=dict(
                # (str) Decay type. Support ['exp', 'linear'].
                type='exp',
                # (float) Epsilon start value.
                start=0.95,
                # (float) Epsilon end value.
                end=0.1,
                # (int) Decay length(env step).
                decay=10000,
            ),
            replay_buffer=dict(
                # (int) Maximum size of replay buffer. Usually, larger buffer size is good.
                replay_buffer_size=10000,
            ),
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

        self._gamma = self._cfg.discount_factor
        self._nstep = self._cfg.nstep
        print(self._cfg)
        self._num_of_model = self._cfg.num_of_model
        
        # build the learn_model_list for averaged_DQN
        if not hasattr(self, '_model_list'):
            self._model_list = [copy.deepcopy(self._model) for _ in range(self._num_of_model)]
        
        # Optimizer
        self._optimizer_list = [Adam(model.parameters(), lr=self._cfg.learn.learning_rate) for model in self._model_list]

        # use model_wrapper for specialized demands of different modes
        self._target_model = copy.deepcopy(self._model)
        if 'target_update_freq' in self._cfg.learn: 
            self._target_model = model_wrap(
                self._target_model,
                wrapper_name='target',
                update_type='assign',
                update_kwargs={'freq': self._cfg.learn.target_update_freq}
            )
        elif 'target_theta' in self._cfg.learn:
            self._target_model = model_wrap(
                self._target_model,
                wrapper_name='target',
                update_type='momentum',
                update_kwargs={'theta': self._cfg.learn.target_theta}
            )
        else:
            raise RuntimeError("DQN needs target network, please either indicate target_update_freq or target_theta")
        self._learn_model_list = [model_wrap(model, wrapper_name='argmax_sample') for model in self._model_list]
        for learn_model in self._learn_model_list:
            learn_model.reset()
        self._target_model.reset()

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
            - optional: ``value_gamma``
        ReturnsKeys:
            - necessary: ``cur_lr``, ``total_loss``, ``priority``
            - optional: ``action_distribution``
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
        # ====================
        # Q-learning forward
        # ====================
        for learn_model in self._learn_model_list:
            learn_model.train()
        self._target_model.train()
        # Current q value of each model
        q_value = 0
        for learn_model in self._learn_model_list:
            q_value += learn_model.forward(data['obs'])['logit']
        q_value /= self._num_of_model
        
        # target q value
        with torch.no_grad():
            target_q_value = self._target_model.forward(data['next_obs'])['logit']
            # Max q value action (main model), i.e. Double DQN
            next_q_value = 0
            for learn_model in self._learn_model_list:
                next_q_value += learn_model.forward(data['next_obs'])['logit']
            next_q_value /= self._num_of_model
            target_q_action = [v.argmax(dim=-1) for v in next_q_value]
            target_q_action = torch.stack(target_q_action).squeeze()

        data_n = q_nstep_td_data(
            q_value, target_q_value, data['action'], target_q_action, data['reward'], data['done'], data['weight']
        )
        value_gamma = data.get('value_gamma')
        loss, td_error_per_sample = q_nstep_td_error(data_n, self._gamma, nstep=self._nstep, value_gamma=value_gamma)

        # ====================
        # Q-learning update
        # ====================
        for optimizer in self._optimizer_list:
            optimizer.zero_grad()
        loss.backward()
        if self._cfg.multi_gpu:
            self.sync_gradients(self._learn_model)
        for optimizer in self._optimizer_list:
            optimizer.step()

        # =============
        # after update
        # =============
        # ?: original paper didn't mention how to update target model
        self._target_model.update(self._learn_model_list[0].state_dict())
        
        return {
            'cur_lr': self._optimizer_list[0].defaults['lr'],
            'total_loss': loss.item(),
            'q_value': q_value.mean().item(),
            'target_q_value': target_q_value.mean().item(),
            'priority': td_error_per_sample.abs().tolist(),
            # Only discrete action satisfying len(data['action'])==1 can return this and draw histogram on tensorboard.
            # '[histogram]action_distribution': data['action'],
        }

    def _init_eval(self) -> None:
        r"""
        Overview:
            Evaluate mode init method. Called by ``self.__init__``, initialize eval_model.
        """
        print(self._cfg)
        self._num_of_model = self._cfg.num_of_model
        if not hasattr(self, '_model_list'):
            self._model_list = [copy.deepcopy(self._model) for _ in range(self._num_of_model)]
        self._eval_model_list = [model_wrap(model, wrapper_name='argmax_sample') for model in self._model_list]
        for eval_model in self._eval_model_list:
            eval_model.reset()

    def _forward_eval(self, data: Dict[int, Any]) -> Dict[int, Any]:
        """
        Overview:
            Forward computation graph of eval mode(evaluate policy performance), at most cases, it is similar to \
            ``self._forward_collect``.
        Arguments:
            - data (:obj:`Dict[str, Any]`): Dict type data, stacked env data for predicting policy_output(action), \
                values are torch.Tensor or np.ndarray or dict/list combinations, keys are env_id indicated by integer.
        Returns:
            - output (:obj:`Dict[int, Any]`): The dict of predicting action for the interaction with env.
        ArgumentsKeys:
            - necessary: ``obs``
        ReturnsKeys
            - necessary: ``action``
        """
        data_id = list(data.keys())
        data = default_collate(list(data.values()))
        if self._cuda:
            data = to_device(data, self._device)
        for eval_model in self._eval_model_list:
            eval_model.eval()
        with torch.no_grad():
            output = self._eval_model_list[0].forward(data)
            q_value = 0
            for eval_model in self._eval_model_list:
                q_value += eval_model.forward(data)['logit']
            q_value /= self._num_of_model
            logit = q_value
            assert isinstance(logit, torch.Tensor) or isinstance(logit, list)
            if isinstance(logit, torch.Tensor):
                logit = [logit]
            output['logit'] = logit
            
            if 'action_mask' in output:
                mask = output['action_mask']
                if isinstance(mask, torch.Tensor):
                    mask = [mask]
                logit = [l.sub_(1e8 * (1 - m)) for l, m in zip(logit, mask)]
            action = [l.argmax(dim=-1) for l in logit]
            if len(action) == 1:
                action, logit = action[0], logit[0]
            output['action'] = action
        
        if self._cuda:
            output = to_device(output, 'cpu')
        output = default_decollate(output)
        return {i: d for i, d in zip(data_id, output)}

    def _state_dict_learn(self) -> Dict[str, Any]:
        """
        Overview:
            Return the state_dict of learn mode, usually including model and optimizer.
        Returns:
            - state_dict (:obj:`Dict[str, Any]`): the dict of current policy learn state, for saving and restoring.
        """
        model_list = [learn_model.state_dict() for learn_model in self._learn_model_list]
        optimizer_list = [optimizer.state_dict() for optimizer in self._optimizer_list]
        return {
            'model_list': model_list,
            'target_model': self._target_model.state_dict(),
            'optimizer_list': optimizer_list,
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
        for idx, learn_model in enumerate(self._learn_model_list):
            learn_model.load_state_dict(state_dict['model_list'][idx])
        self._target_model.load_state_dict(state_dict['target_model'])
        for idx, optimizer in enumerate(self._optimizer_list):
            optimizer.load_state_dict(state_dict['optimizer_list'][idx])