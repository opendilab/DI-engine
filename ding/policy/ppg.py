from typing import List, Dict, Any, Tuple, Union
from collections import namedtuple
import copy
import torch
from torch.utils.data import Dataset, DataLoader

from ding.utils import POLICY_REGISTRY, split_data_generator, RunningMeanStd
from ding.utils.data import default_collate, default_decollate
from ding.torch_utils import Adam, to_device
from ding.rl_utils import get_gae_with_default_last_value, get_train_sample, gae, gae_data, get_gae, \
    ppo_policy_data, ppo_policy_error, ppo_value_data, ppo_value_error, ppg_data, ppg_joint_error
from ding.model import model_wrap
from .base_policy import Policy


class ExperienceDataset(Dataset):

    def __init__(self, data):
        super().__init__()
        self.data = data

    def __len__(self):
        return list(self.data.values())[0].shape[0]

    def __getitem__(self, ind):
        data = {}
        for key in self.data.keys():
            data[key] = self.data[key][ind]
        return data


def create_shuffled_dataloader(data, batch_size):
    ds = ExperienceDataset(data)
    return DataLoader(ds, batch_size=batch_size, shuffle=True)


@POLICY_REGISTRY.register('ppg')
class PPGPolicy(Policy):
    """
    Overview:
        Policy class of PPG algorithm.
    Interface:
        _init_learn, _data_preprocess_learn, _forward_learn, _state_dict_learn, _load_state_dict_learn\
            _init_collect, _forward_collect, _process_transition, _get_train_sample, _get_batch_size, _init_eval,\
            _forward_eval, default_model, _monitor_vars_learn, learn_aux
    Config:
        == ==================== ======== ============== ======================================== =======================
        ID Symbol               Type     Default Value  Description                              Other(Shape)
        == ==================== ======== ============== ======================================== =======================
        1  ``type``             str      ppg            | RL policy register name, refer to      | this arg is optional,
                                                        | registry ``POLICY_REGISTRY``           | a placeholder
        2  ``cuda``             bool     False          | Whether to use cuda for network        | this arg can be diff-
                                                                                                 | erent from modes
        3  ``on_policy``        bool     True           | Whether the RL algorithm is on-policy
                                                        | or off-policy
        4. ``priority``         bool     False          | Whether use priority(PER)              | priority sample,
                                                                                                 | update priority
        5  | ``priority_``      bool     False          | Whether use Importance Sampling        | IS weight
           | ``IS_weight``                              | Weight to correct biased update.
        6  | ``learn.update``   int      5              | How many updates(iterations) to train  | this args can be vary
           | ``_per_collect``                           | after collector's one collection. Only | from envs. Bigger val
                                                        | valid in serial training               | means more off-policy
        7  | ``learn.value_``   float    1.0            | The loss weight of value network       | policy network weight
           | ``weight``                                                                          | is set to 1
        8  | ``learn.entropy_`` float    0.01           | The loss weight of entropy             | policy network weight
           | ``weight``                                 | regularization                         | is set to 1
        9  | ``learn.clip_``    float    0.2            | PPO clip ratio
           | ``ratio``
        10 | ``learn.adv_``     bool     False          | Whether to use advantage norm in
           | ``norm``                                   | a whole training batch
        11 | ``learn.aux_``     int      5              | The frequency(normal update times)
           | ``freq``                                   | of auxiliary phase training
        12 | ``learn.aux_``     int      6              | The training epochs of auxiliary
           | ``train_epoch``                            | phase
        13 | ``learn.aux_``     int      1              | The loss weight of behavioral_cloning
           | ``bc_weight``                              | in auxiliary phase
        14 | ``collect.dis``    float    0.99           | Reward's future discount factor, aka.  | may be 1 when sparse
           | ``count_factor``                           | gamma                                  | reward env
        15 | ``collect.gae_``   float    0.95           | GAE lambda factor for the balance
           | ``lambda``                                 | of bias and variance(1-step td and mc)
        == ==================== ======== ============== ======================================== =======================
    """
    config = dict(
        # (str) RL policy register name (refer to function "POLICY_REGISTRY").
        type='ppg',
        # (bool) Whether to use cuda for network.
        cuda=False,
        # (bool) Whether the RL algorithm is on-policy or off-policy. (Note: in practice PPO can be off-policy used)
        on_policy=True,
        priority=False,
        # (bool) Whether use Importance Sampling Weight to correct biased update. If True, priority must be True.
        priority_IS_weight=False,
        learn=dict(
            # (bool) Whether to use multi gpu
            multi_gpu=False,
            actor_epoch_per_collect=1,
            critic_epoch_per_collect=1,
            batch_size=64,
            learning_rate=0.001,
            # ==============================================================
            # The following configs is algorithm-specific
            # ==============================================================
            # (float) The loss weight of value network, policy network weight is set to 1
            value_weight=0.5,
            # (float) The loss weight of entropy regularization, policy network weight is set to 1
            entropy_weight=0.01,
            # (float) PPO clip ratio, defaults to 0.2
            clip_ratio=0.2,
            value_norm=False,
            # (bool) Whether to use advantage norm in a whole training batch
            adv_norm=False,
            # (int) The frequency(normal update times) of auxiliary phase training
            aux_freq=8,
            # (int) The training epochs of auxiliary phase
            aux_train_epoch=6,
            # (int) The loss weight of behavioral_cloning in auxiliary phase
            aux_bc_weight=1,
            grad_clip_type='clip_norm',
            grad_clip_value=10,
            ignore_done=False,
        ),
        collect=dict(
            # n_sample=64,
            unroll_len=1,
            # ==============================================================
            # The following configs is algorithm-specific
            # ==============================================================
            # (float) Reward's future discount factor, aka. gamma.
            discount_factor=0.99,
            # (float) GAE lambda factor for the balance of bias and variance(1-step td and mc)
            gae_lambda=0.95,
        ),
        eval=dict(),
    )

    def _init_learn(self) -> None:
        r"""
        Overview:
            Learn mode init method. Called by ``self.__init__``.
            Init the optimizer, algorithm config and the main model.
        Arguments:
            .. note::

                The _init_learn method takes the argument from the self._cfg.learn in the config file

            - learning_rate (:obj:`float`): The learning rate fo the optimizer
        """
        # Optimizer
        self._optimizer_ac = Adam(self._model.actor_critic.parameters(), lr=self._cfg.learn.learning_rate)
        self._optimizer_aux_critic = Adam(self._model.aux_critic.parameters(), lr=self._cfg.learn.learning_rate)
        self._learn_model = model_wrap(self._model, wrapper_name='base')

        # Algorithm config
        self._priority = self._cfg.priority
        self._priority_IS_weight = self._cfg.priority_IS_weight
        assert not self._priority and not self._priority_IS_weight, "Priority is not implemented in PPG"
        self._value_weight = self._cfg.learn.value_weight
        self._entropy_weight = self._cfg.learn.entropy_weight
        self._value_norm = self._cfg.learn.value_norm
        if self._value_norm:
            self._running_mean_std = RunningMeanStd(epsilon=1e-4, device=self._device)
        self._clip_ratio = self._cfg.learn.clip_ratio
        self._adv_norm = self._cfg.learn.adv_norm

        # Main model
        self._learn_model.reset()

        # Auxiliary memories
        self._aux_train_epoch = self._cfg.learn.aux_train_epoch
        self._train_iteration = 0
        self._aux_memories = []
        self._aux_bc_weight = self._cfg.learn.aux_bc_weight

    def _data_preprocess_learn(self, data: List[Any]) -> dict:
        """
        Overview:
            Preprocess the data to fit the required data format for learning, including \
            collate(stack data into batch), ignore done(in some fake terminate env),\
            prepare loss weight per training sample, and cpu tensor to cuda.
        Arguments:
            - data (:obj:`List[Dict[str, Any]]`): the data collected from collect function
        Returns:
            - data (:obj:`Dict[str, Any]`): the processed data, including at least ['done', 'weight']
        """
        # data preprocess
        data = default_collate(data)
        ignore_done = self._cfg.learn.ignore_done
        if ignore_done:
            data['done'] = None
        else:
            data['done'] = data['done'].float()
        data['weight'] = None
        if self._cuda:
            data = to_device(data, self._device)
        return data

    def _forward_learn(self, data: dict) -> Dict[str, Any]:
        """
        Overview:
            Forward and backward function of learn mode.
        Arguments:
            - data (:obj:`Dict[str, Any]`): Dict type data, a batch of data for training, values are torch.Tensor or \
                np.ndarray or dict/list combinations.
        Returns:
            - info_dict (:obj:`Dict[str, Any]`): Dict type data, a info dict indicated training result, which will be \
                recorded in text log and tensorboard, values are python scalar or a list of scalars.
        ArgumentsKeys:
            - necessary: 'obs', 'logit', 'action', 'value', 'reward', 'done'
        ReturnsKeys:
            - necessary: current lr, total_loss, policy_loss, value_loss, entropy_loss, \
                        adv_abs_max, approx_kl, clipfrac\
                        aux_value_loss, auxiliary_loss, behavioral_cloning_loss

                - current_lr (:obj:`float`): Current learning rate
                - total_loss (:obj:`float`): The calculated loss
                - policy_loss (:obj:`float`): The policy(actor) loss of ppg
                - value_loss (:obj:`float`): The value(critic) loss of ppg
                - entropy_loss (:obj:`float`): The entropy loss
                - auxiliary_loss (:obj:`float`): The auxiliary loss, we use the value function loss \
                    as the auxiliary objective, thereby sharing features between the policy and value function\
                    while minimizing distortions to the policy
                - aux_value_loss (:obj:`float`): The auxiliary value loss, we need to train the value network extra \
                    during the auxiliary phase, it's the value loss we train the value network during auxiliary phase
                - behavioral_cloning_loss (:obj:`float`): The behavioral cloning loss, used to optimize the auxiliary\
                     objective while otherwise preserving the original policy
        """
        data = self._data_preprocess_learn(data)
        # ====================
        # PPG forward
        # ====================
        self._learn_model.train()
        return_infos = []
        if self._value_norm:
            unnormalized_return = data['adv'] + data['value'] * self._running_mean_std.std
            data['return'] = unnormalized_return / self._running_mean_std.std
            self._running_mean_std.update(unnormalized_return.cpu().numpy())
        else:
            data['return'] = data['adv'] + data['value']

        for epoch in range(self._cfg.learn.actor_epoch_per_collect):
            for policy_data in split_data_generator(data, self._cfg.learn.batch_size, shuffle=True):
                policy_adv = policy_data['adv']
                if self._adv_norm:
                    # Normalize advantage in a total train_batch
                    policy_adv = (policy_adv - policy_adv.mean()) / (policy_adv.std() + 1e-8)
                # Policy Phase(Policy)
                policy_output = self._learn_model.forward(policy_data['obs'], mode='compute_actor')
                policy_error_data = ppo_policy_data(
                    policy_output['logit'], policy_data['logit'], policy_data['action'], policy_adv,
                    policy_data['weight']
                )
                ppo_policy_loss, ppo_info = ppo_policy_error(policy_error_data, self._clip_ratio)
                policy_loss = ppo_policy_loss.policy_loss - self._entropy_weight * ppo_policy_loss.entropy_loss
                self._optimizer_ac.zero_grad()
                policy_loss.backward()
                self._optimizer_ac.step()

        for epoch in range(self._cfg.learn.critic_epoch_per_collect):
            for value_data in split_data_generator(data, self._cfg.learn.batch_size, shuffle=True):
                value_adv = value_data['adv']
                return_ = value_data['return']
                if self._adv_norm:
                    # Normalize advantage in a total train_batch
                    value_adv = (value_adv - value_adv.mean()) / (value_adv.std() + 1e-8)
                # Policy Phase(Value)
                value_output = self._learn_model.forward(value_data['obs'], mode='compute_critic')
                value_error_data = ppo_value_data(
                    value_output['value'], value_data['value'], return_, value_data['weight']
                )
                value_loss = self._value_weight * ppo_value_error(value_error_data, self._clip_ratio)
                self._optimizer_aux_critic.zero_grad()
                value_loss.backward()
                self._optimizer_aux_critic.step()

        data['return_'] = data['return']

        self._aux_memories.append(copy.deepcopy(data))

        self._train_iteration += 1

        # ====================
        # PPG update
        # use aux loss after iterations and reset aux_memories
        # ====================

        # Auxiliary Phase
        # record data for auxiliary head

        if self._train_iteration % self._cfg.learn.aux_freq == 0:
            aux_loss, bc_loss, aux_value_loss = self.learn_aux()
            return {
                'policy_cur_lr': self._optimizer_ac.defaults['lr'],
                'value_cur_lr': self._optimizer_aux_critic.defaults['lr'],
                'policy_loss': ppo_policy_loss.policy_loss.item(),
                'value_loss': value_loss.item(),
                'entropy_loss': ppo_policy_loss.entropy_loss.item(),
                'policy_adv_abs_max': policy_adv.abs().max().item(),
                'approx_kl': ppo_info.approx_kl,
                'clipfrac': ppo_info.clipfrac,
                'aux_value_loss': aux_value_loss,
                'auxiliary_loss': aux_loss,
                'behavioral_cloning_loss': bc_loss,
            }
        else:
            return {
                'policy_cur_lr': self._optimizer_ac.defaults['lr'],
                'value_cur_lr': self._optimizer_aux_critic.defaults['lr'],
                'policy_loss': ppo_policy_loss.policy_loss.item(),
                'value_loss': value_loss.item(),
                'entropy_loss': ppo_policy_loss.entropy_loss.item(),
                'policy_adv_abs_max': policy_adv.abs().max().item(),
                'approx_kl': ppo_info.approx_kl,
                'clipfrac': ppo_info.clipfrac,
            }

    def _state_dict_learn(self) -> Dict[str, Any]:
        r"""
        Overview:
            Return the state_dict of learn mode, usually including model and optimizer.
        Returns:
            - state_dict (:obj:`Dict[str, Any]`): the dict of current policy learn state, for saving and restoring.
        """
        return {
            'model': self._learn_model.state_dict(),
            'optimizer_ac': self._optimizer_ac.state_dict(),
            'optimizer_aux_critic': self._optimizer_aux_critic.state_dict(),
        }

    def _load_state_dict_learn(self, state_dict: Dict[str, Any]) -> None:
        """
        Overview:
            Load the state_dict variable into policy learn mode.
        Arguments:
            - state_dict (:obj:`Dict[str, Any]`): the dict of policy learn state saved before.\
                When the value is distilled into the policy network, we need to make sure the policy \
                network does not change the action predictions, we need two optimizers, \
                _optimizer_ac is used in policy net, and _optimizer_aux_critic is used in value net.

        .. tip::
            If you want to only load some parts of model, you can simply set the ``strict`` argument in \
            load_state_dict to ``False``, or refer to ``ding.torch_utils.checkpoint_helper`` for more \
            complicated operation.
        """
        self._learn_model.load_state_dict(state_dict['model'])
        self._optimizer_ac.load_state_dict(state_dict['optimizer_ac'])
        self._optimizer_aux_critic.load_state_dict(state_dict['optimizer_aux_critic'])

    def _init_collect(self) -> None:
        r"""
        Overview:
            Collect mode init method. Called by ``self.__init__``.
            Init unroll length, collect model.
        """
        self._unroll_len = self._cfg.collect.unroll_len
        self._collect_model = model_wrap(self._model, wrapper_name='multinomial_sample')
        # TODO continuous action space exploration
        self._collect_model.reset()
        self._gamma = self._cfg.collect.discount_factor
        self._gae_lambda = self._cfg.collect.gae_lambda

    def _forward_collect(self, data: dict) -> dict:
        r"""
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
        self._collect_model.eval()
        with torch.no_grad():
            output = self._collect_model.forward(data, mode='compute_actor_critic')
        if self._cuda:
            output = to_device(output, 'cpu')
        output = default_decollate(output)
        return {i: d for i, d in zip(data_id, output)}

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
            'logit': model_output['logit'],
            'action': model_output['action'],
            'value': model_output['value'],
            'reward': timestep.reward,
            'done': timestep.done,
        }
        return transition

    def _get_train_sample(self, data: list) -> Union[None, List[Any]]:
        r"""
        Overview:
            Get the trajectory and calculate GAE, return one data to cache for next time calculation
        Arguments:
            - data (:obj:`list`): The trajectory's cache
        Returns:
            - samples (:obj:`dict`): The training samples generated
        """
        data = to_device(data, self._device)
        if self._cfg.learn.ignore_done:
            data[-1]['done'] = False

        if data[-1]['done']:
            last_value = torch.zeros_like(data[-1]['value'])
        else:
            with torch.no_grad():
                last_value = self._collect_model.forward(
                    data[-1]['next_obs'].unsqueeze(0), mode='compute_actor_critic'
                )['value']
        if self._value_norm:
            last_value *= self._running_mean_std.std
            for i in range(len(data)):
                data[i]['value'] *= self._running_mean_std.std
        data = get_gae(
            data,
            to_device(last_value, self._device),
            gamma=self._gamma,
            gae_lambda=self._gae_lambda,
            cuda=False,
        )
        if self._value_norm:
            for i in range(len(data)):
                data[i]['value'] /= self._running_mean_std.std

        return get_train_sample(data, self._unroll_len)

    def _get_batch_size(self) -> Dict[str, int]:
        """
        Overview:
            Get learn batch size. In the PPG algorithm, different networks require different data.\
            We need to get data['policy'] and data['value'] to train policy net and value net,\
            this function is used to get the batch size of data['policy'] and data['value'].
        Returns:
            - output (:obj:`dict[str, int]`): Dict type data, including str type batch size and int type batch size.
        """
        bs = self._cfg.learn.batch_size
        return {'policy': bs, 'value': bs}

    def _init_eval(self) -> None:
        r"""
        Overview:
            Evaluate mode init method. Called by ``self.__init__``.
            Init eval model with argmax strategy.
        """
        self._eval_model = model_wrap(self._model, wrapper_name='argmax_sample')
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
            output = self._eval_model.forward(data, mode='compute_actor')
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
            by import_names path.
        """
        return 'ppg', ['ding.model.template.ppg']

    def _monitor_vars_learn(self) -> List[str]:
        r"""
        Overview:
            Return variables' name if variables are to used in monitor.
        Returns:
            - vars (:obj:`List[str]`): Variables' name list.
        """
        return [
            'policy_cur_lr',
            'value_cur_lr',
            'policy_loss',
            'value_loss',
            'entropy_loss',
            'policy_adv_abs_max',
            'approx_kl',
            'clipfrac',
            'aux_value_loss',
            'auxiliary_loss',
            'behavioral_cloning_loss',
        ]

    def learn_aux(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Overview:
            The auxiliary phase training, where the value is distilled into the policy network
        Returns:
            - aux_loss (:obj:`Tuple[torch.Tensor, torch.Tensor, torch.Tensor]`): including average auxiliary loss\
                average behavioral cloning loss, and average auxiliary value loss
        """
        aux_memories = self._aux_memories
        # gather states and target values into one tensor
        data = {}
        states = []
        actions = []
        return_ = []
        old_values = []
        weights = []
        for memory in aux_memories:
            # for memory in memories:
            states.append(memory['obs'])
            actions.append(memory['action'])
            return_.append(memory['return_'])
            old_values.append(memory['value'])
            if memory['weight'] is None:
                weight = torch.ones_like(memory['action'])
            else:
                weight = torch.tensor(memory['weight'])
            weights.append(weight)

        data['obs'] = torch.cat(states)
        data['action'] = torch.cat(actions)
        data['return_'] = torch.cat(return_)
        data['value'] = torch.cat(old_values)
        data['weight'] = torch.cat(weights).float()
        # compute current policy logit_old
        with torch.no_grad():
            data['logit_old'] = self._model.forward(data['obs'], mode='compute_actor')['logit']

        # prepared dataloader for auxiliary phase training
        dl = create_shuffled_dataloader(data, self._cfg.learn.batch_size)

        # the proposed auxiliary phase training
        # where the value is distilled into the policy network,
        # while making sure the policy network does not change the action predictions (kl div loss)

        i = 0
        auxiliary_loss_ = 0
        behavioral_cloning_loss_ = 0
        value_loss_ = 0

        for epoch in range(self._aux_train_epoch):
            for data in dl:
                policy_output = self._model.forward(data['obs'], mode='compute_actor_critic')

                # Calculate ppg error 'logit_new', 'logit_old', 'action', 'value_new', 'value_old', 'return_', 'weight'
                data_ppg = ppg_data(
                    policy_output['logit'], data['logit_old'], data['action'], policy_output['value'], data['value'],
                    data['return_'], data['weight']
                )
                ppg_joint_loss = ppg_joint_error(data_ppg, self._clip_ratio)
                wb = self._aux_bc_weight
                total_loss = ppg_joint_loss.auxiliary_loss + wb * ppg_joint_loss.behavioral_cloning_loss

                # # policy network loss copmoses of both the kl div loss as well as the auxiliary loss
                # aux_loss = clipped_value_loss(policy_values, rewards, old_values, self.value_clip)
                # loss_kl = F.kl_div(action_logprobs, old_action_probs, reduction='batchmean')
                # policy_loss = aux_loss + loss_kl

                self._optimizer_ac.zero_grad()
                total_loss.backward()
                self._optimizer_ac.step()

                # paper says it is important to train the value network extra during the auxiliary phase
                # Calculate ppg error 'value_new', 'value_old', 'return_', 'weight'
                values = self._model.forward(data['obs'], mode='compute_critic')['value']
                data_aux = ppo_value_data(values, data['value'], data['return_'], data['weight'])

                value_loss = ppo_value_error(data_aux, self._clip_ratio)

                self._optimizer_aux_critic.zero_grad()
                value_loss.backward()
                self._optimizer_aux_critic.step()

                auxiliary_loss_ += ppg_joint_loss.auxiliary_loss.item()
                behavioral_cloning_loss_ += ppg_joint_loss.behavioral_cloning_loss.item()
                value_loss_ += value_loss.item()
                i += 1

        self._aux_memories = []

        return auxiliary_loss_ / i, behavioral_cloning_loss_ / i, value_loss_ / i


@POLICY_REGISTRY.register('ppg_offpolicy')
class PPGOffPolicy(Policy):
    """
    Overview:
        Policy class of PPG algorithm.
    Interface:
        _init_learn, _data_preprocess_learn, _forward_learn, _state_dict_learn, _load_state_dict_learn\
            _init_collect, _forward_collect, _process_transition, _get_train_sample, _get_batch_size, _init_eval,\
            _forward_eval, default_model, _monitor_vars_learn, learn_aux
    Config:
        == ==================== ======== ============== ======================================== =======================
        ID Symbol               Type     Default Value  Description                              Other(Shape)
        == ==================== ======== ============== ======================================== =======================
        1  ``type``             str      ppg            | RL policy register name, refer to      | this arg is optional,
                                                        | registry ``POLICY_REGISTRY``           | a placeholder
        2  ``cuda``             bool     False          | Whether to use cuda for network        | this arg can be diff-
                                                                                                 | erent from modes
        3  ``on_policy``        bool     True           | Whether the RL algorithm is on-policy
                                                        | or off-policy
        4. ``priority``         bool     False          | Whether use priority(PER)              | priority sample,
                                                                                                 | update priority
        5  | ``priority_``      bool     False          | Whether use Importance Sampling        | IS weight
           | ``IS_weight``                              | Weight to correct biased update.
        6  | ``learn.update``   int      5              | How many updates(iterations) to train  | this args can be vary
           | ``_per_collect``                           | after collector's one collection. Only | from envs. Bigger val
                                                        | valid in serial training               | means more off-policy
        7  | ``learn.value_``   float    1.0            | The loss weight of value network       | policy network weight
           | ``weight``                                                                          | is set to 1
        8  | ``learn.entropy_`` float    0.01           | The loss weight of entropy             | policy network weight
           | ``weight``                                 | regularization                         | is set to 1
        9  | ``learn.clip_``    float    0.2            | PPO clip ratio
           | ``ratio``
        10 | ``learn.adv_``     bool     False          | Whether to use advantage norm in
           | ``norm``                                   | a whole training batch
        11 | ``learn.aux_``     int      5              | The frequency(normal update times)
           | ``freq``                                   | of auxiliary phase training
        12 | ``learn.aux_``     int      6              | The training epochs of auxiliary
           | ``train_epoch``                            | phase
        13 | ``learn.aux_``     int      1              | The loss weight of behavioral_cloning
           | ``bc_weight``                              | in auxiliary phase
        14 | ``collect.dis``    float    0.99           | Reward's future discount factor, aka.  | may be 1 when sparse
           | ``count_factor``                           | gamma                                  | reward env
        15 | ``collect.gae_``   float    0.95           | GAE lambda factor for the balance
           | ``lambda``                                 | of bias and variance(1-step td and mc)
        == ==================== ======== ============== ======================================== =======================
    """
    config = dict(
        # (str) RL policy register name (refer to function "POLICY_REGISTRY").
        type='ppg_offpolicy',
        # (bool) Whether to use cuda for network.
        cuda=False,
        # (bool) Whether the RL algorithm is on-policy or off-policy. (Note: in practice PPO can be off-policy used)
        on_policy=False,
        priority=False,
        # (bool) Whether use Importance Sampling Weight to correct biased update. If True, priority must be True.
        priority_IS_weight=False,
        # (bool) Whether to need policy data in process transition
        transition_with_policy_data=True,
        learn=dict(
            # (bool) Whether to use multi gpu
            multi_gpu=False,
            update_per_collect=5,
            batch_size=64,
            learning_rate=0.001,
            # ==============================================================
            # The following configs is algorithm-specific
            # ==============================================================
            # (float) The loss weight of value network, policy network weight is set to 1
            value_weight=0.5,
            # (float) The loss weight of entropy regularization, policy network weight is set to 1
            entropy_weight=0.01,
            # (float) PPO clip ratio, defaults to 0.2
            clip_ratio=0.2,
            # (bool) Whether to use advantage norm in a whole training batch
            adv_norm=False,
            # (int) The frequency(normal update times) of auxiliary phase training
            aux_freq=5,
            # (int) The training epochs of auxiliary phase
            aux_train_epoch=6,
            # (int) The loss weight of behavioral_cloning in auxiliary phase
            aux_bc_weight=1,
            ignore_done=False,
        ),
        collect=dict(
            # n_sample=64,
            unroll_len=1,
            # ==============================================================
            # The following configs is algorithm-specific
            # ==============================================================
            # (float) Reward's future discount factor, aka. gamma.
            discount_factor=0.99,
            # (float) GAE lambda factor for the balance of bias and variance(1-step td and mc)
            gae_lambda=0.95,
        ),
        eval=dict(),
        other=dict(
            replay_buffer=dict(
                # PPG use two separate buffer for different reuse
                multi_buffer=True,
                policy=dict(replay_buffer_size=1000, ),
                value=dict(replay_buffer_size=1000, ),
            ),
        ),
    )

    def _init_learn(self) -> None:
        r"""
        Overview:
            Learn mode init method. Called by ``self.__init__``.
            Init the optimizer, algorithm config and the main model.
        Arguments:
            .. note::

                The _init_learn method takes the argument from the self._cfg.learn in the config file

            - learning_rate (:obj:`float`): The learning rate fo the optimizer
        """
        # Optimizer
        self._optimizer_ac = Adam(self._model.actor_critic.parameters(), lr=self._cfg.learn.learning_rate)
        self._optimizer_aux_critic = Adam(self._model.aux_critic.parameters(), lr=self._cfg.learn.learning_rate)
        self._learn_model = model_wrap(self._model, wrapper_name='base')

        # Algorithm config
        self._priority = self._cfg.priority
        self._priority_IS_weight = self._cfg.priority_IS_weight
        assert not self._priority and not self._priority_IS_weight, "Priority is not implemented in PPG"
        self._value_weight = self._cfg.learn.value_weight
        self._entropy_weight = self._cfg.learn.entropy_weight
        self._clip_ratio = self._cfg.learn.clip_ratio
        self._adv_norm = self._cfg.learn.adv_norm

        # Main model
        self._learn_model.reset()

        # Auxiliary memories
        self._aux_train_epoch = self._cfg.learn.aux_train_epoch
        self._train_iteration = 0
        self._aux_memories = []
        self._aux_bc_weight = self._cfg.learn.aux_bc_weight

    def _data_preprocess_learn(self, data: List[Any]) -> dict:
        """
        Overview:
            Preprocess the data to fit the required data format for learning, including \
            collate(stack data into batch), ignore done(in some fake terminate env),\
            prepare loss weight per training sample, and cpu tensor to cuda.
        Arguments:
            - data (:obj:`List[Dict[str, Any]]`): the data collected from collect function
        Returns:
            - data (:obj:`Dict[str, Any]`): the processed data, including at least ['done', 'weight']
        """
        # data preprocess
        for k, data_item in data.items():
            data_item = default_collate(data_item)
            ignore_done = self._cfg.learn.ignore_done
            if ignore_done:
                data_item['done'] = None
            else:
                data_item['done'] = data_item['done'].float()
            data_item['weight'] = None
            data[k] = data_item
        if self._cuda:
            data = to_device(data, self._device)
        return data

    def _forward_learn(self, data: dict) -> Dict[str, Any]:
        """
        Overview:
            Forward and backward function of learn mode.
        Arguments:
            - data (:obj:`Dict[str, Any]`): Dict type data, a batch of data for training, values are torch.Tensor or \
                np.ndarray or dict/list combinations.
        Returns:
            - info_dict (:obj:`Dict[str, Any]`): Dict type data, a info dict indicated training result, which will be \
                recorded in text log and tensorboard, values are python scalar or a list of scalars.
        ArgumentsKeys:
            - necessary: 'obs', 'logit', 'action', 'value', 'reward', 'done'
        ReturnsKeys:
            - necessary: current lr, total_loss, policy_loss, value_loss, entropy_loss, \
                        adv_abs_max, approx_kl, clipfrac\
                        aux_value_loss, auxiliary_loss, behavioral_cloning_loss

                - current_lr (:obj:`float`): Current learning rate
                - total_loss (:obj:`float`): The calculated loss
                - policy_loss (:obj:`float`): The policy(actor) loss of ppg
                - value_loss (:obj:`float`): The value(critic) loss of ppg
                - entropy_loss (:obj:`float`): The entropy loss
                - auxiliary_loss (:obj:`float`): The auxiliary loss, we use the value function loss \
                    as the auxiliary objective, thereby sharing features between the policy and value function\
                    while minimizing distortions to the policy
                - aux_value_loss (:obj:`float`): The auxiliary value loss, we need to train the value network extra \
                    during the auxiliary phase, it's the value loss we train the value network during auxiliary phase
                - behavioral_cloning_loss (:obj:`float`): The behavioral cloning loss, used to optimize the auxiliary\
                     objective while otherwise preserving the original policy
        """
        data = self._data_preprocess_learn(data)
        # ====================
        # PPG forward
        # ====================
        self._learn_model.train()
        policy_data, value_data = data['policy'], data['value']
        policy_adv, value_adv = policy_data['adv'], value_data['adv']
        return_ = value_data['value'] + value_adv
        if self._adv_norm:
            # Normalize advantage in a total train_batch
            policy_adv = (policy_adv - policy_adv.mean()) / (policy_adv.std() + 1e-8)
            value_adv = (value_adv - value_adv.mean()) / (value_adv.std() + 1e-8)
        # Policy Phase(Policy)
        policy_output = self._learn_model.forward(policy_data['obs'], mode='compute_actor')
        policy_error_data = ppo_policy_data(
            policy_output['logit'], policy_data['logit'], policy_data['action'], policy_adv, policy_data['weight']
        )
        ppo_policy_loss, ppo_info = ppo_policy_error(policy_error_data, self._clip_ratio)
        policy_loss = ppo_policy_loss.policy_loss - self._entropy_weight * ppo_policy_loss.entropy_loss
        self._optimizer_ac.zero_grad()
        policy_loss.backward()
        self._optimizer_ac.step()

        # Policy Phase(Value)
        value_output = self._learn_model.forward(value_data['obs'], mode='compute_critic')
        value_error_data = ppo_value_data(value_output['value'], value_data['value'], return_, value_data['weight'])
        value_loss = self._value_weight * ppo_value_error(value_error_data, self._clip_ratio)
        self._optimizer_aux_critic.zero_grad()
        value_loss.backward()
        self._optimizer_aux_critic.step()

        # ====================
        # PPG update
        # use aux loss after iterations and reset aux_memories
        # ====================

        # Auxiliary Phase
        # record data for auxiliary head
        data = data['value']
        data['return_'] = return_.data
        self._aux_memories.append(copy.deepcopy(data))

        self._train_iteration += 1
        total_loss = policy_loss + value_loss
        if self._train_iteration % self._cfg.learn.aux_freq == 0:
            aux_loss, bc_loss, aux_value_loss = self.learn_aux()
            total_loss += aux_loss + bc_loss + aux_value_loss
            return {
                'policy_cur_lr': self._optimizer_ac.defaults['lr'],
                'value_cur_lr': self._optimizer_aux_critic.defaults['lr'],
                'policy_loss': ppo_policy_loss.policy_loss.item(),
                'value_loss': value_loss.item(),
                'entropy_loss': ppo_policy_loss.entropy_loss.item(),
                'policy_adv_abs_max': policy_adv.abs().max().item(),
                'approx_kl': ppo_info.approx_kl,
                'clipfrac': ppo_info.clipfrac,
                'aux_value_loss': aux_value_loss,
                'auxiliary_loss': aux_loss,
                'behavioral_cloning_loss': bc_loss,
                'total_loss': total_loss.item(),
            }
        else:
            return {
                'policy_cur_lr': self._optimizer_ac.defaults['lr'],
                'value_cur_lr': self._optimizer_aux_critic.defaults['lr'],
                'policy_loss': ppo_policy_loss.policy_loss.item(),
                'value_loss': value_loss.item(),
                'entropy_loss': ppo_policy_loss.entropy_loss.item(),
                'policy_adv_abs_max': policy_adv.abs().max().item(),
                'approx_kl': ppo_info.approx_kl,
                'clipfrac': ppo_info.clipfrac,
                'total_loss': total_loss.item(),
            }

    def _state_dict_learn(self) -> Dict[str, Any]:
        r"""
        Overview:
            Return the state_dict of learn mode, usually including model and optimizer.
        Returns:
            - state_dict (:obj:`Dict[str, Any]`): the dict of current policy learn state, for saving and restoring.
        """
        return {
            'model': self._learn_model.state_dict(),
            'optimizer_ac': self._optimizer_ac.state_dict(),
            'optimizer_aux_critic': self._optimizer_aux_critic.state_dict(),
        }

    def _load_state_dict_learn(self, state_dict: Dict[str, Any]) -> None:
        """
        Overview:
            Load the state_dict variable into policy learn mode.
        Arguments:
            - state_dict (:obj:`Dict[str, Any]`): the dict of policy learn state saved before.\
                When the value is distilled into the policy network, we need to make sure the policy \
                network does not change the action predictions, we need two optimizers, \
                _optimizer_ac is used in policy net, and _optimizer_aux_critic is used in value net.

        .. tip::
            If you want to only load some parts of model, you can simply set the ``strict`` argument in \
            load_state_dict to ``False``, or refer to ``ding.torch_utils.checkpoint_helper`` for more \
            complicated operation.
        """
        self._learn_model.load_state_dict(state_dict['model'])
        self._optimizer_ac.load_state_dict(state_dict['optimizer_ac'])
        self._optimizer_aux_critic.load_state_dict(state_dict['optimizer_aux_critic'])

    def _init_collect(self) -> None:
        r"""
        Overview:
            Collect mode init method. Called by ``self.__init__``.
            Init unroll length, collect model.
        """
        self._unroll_len = self._cfg.collect.unroll_len
        self._collect_model = model_wrap(self._model, wrapper_name='multinomial_sample')
        # TODO continuous action space exploration
        self._collect_model.reset()
        self._gamma = self._cfg.collect.discount_factor
        self._gae_lambda = self._cfg.collect.gae_lambda

    def _forward_collect(self, data: dict) -> dict:
        r"""
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
        self._collect_model.eval()
        with torch.no_grad():
            output = self._collect_model.forward(data, mode='compute_actor_critic')
        if self._cuda:
            output = to_device(output, 'cpu')
        output = default_decollate(output)
        return {i: d for i, d in zip(data_id, output)}

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
            'logit': model_output['logit'],
            'action': model_output['action'],
            'value': model_output['value'],
            'reward': timestep.reward,
            'done': timestep.done,
        }
        return transition

    def _get_train_sample(self, data: list) -> Union[None, List[Any]]:
        r"""
        Overview:
            Get the trajectory and calculate GAE, return one data to cache for next time calculation
        Arguments:
            - data (:obj:`list`): The trajectory's cache
        Returns:
            - samples (:obj:`dict`): The training samples generated
        """
        data = get_gae_with_default_last_value(
            data,
            data[-1]['done'],
            gamma=self._gamma,
            gae_lambda=self._gae_lambda,
            cuda=False,
        )
        data = get_train_sample(data, self._unroll_len)
        for d in data:
            d['buffer_name'] = ["policy", "value"]
        return data

    def _get_batch_size(self) -> Dict[str, int]:
        """
        Overview:
            Get learn batch size. In the PPG algorithm, different networks require different data.\
            We need to get data['policy'] and data['value'] to train policy net and value net,\
            this function is used to get the batch size of data['policy'] and data['value'].
        Returns:
            - output (:obj:`dict[str, int]`): Dict type data, including str type batch size and int type batch size.
        """
        bs = self._cfg.learn.batch_size
        return {'policy': bs, 'value': bs}

    def _init_eval(self) -> None:
        r"""
        Overview:
            Evaluate mode init method. Called by ``self.__init__``.
            Init eval model with argmax strategy.
        """
        self._eval_model = model_wrap(self._model, wrapper_name='argmax_sample')
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
            output = self._eval_model.forward(data, mode='compute_actor')
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
            by import_names path.
        """
        return 'ppg', ['ding.model.template.ppg']

    def _monitor_vars_learn(self) -> List[str]:
        r"""
        Overview:
            Return variables' name if variables are to used in monitor.
        Returns:
            - vars (:obj:`List[str]`): Variables' name list.
        """
        return [
            'policy_cur_lr',
            'value_cur_lr',
            'policy_loss',
            'value_loss',
            'entropy_loss',
            'policy_adv_abs_max',
            'approx_kl',
            'clipfrac',
            'aux_value_loss',
            'auxiliary_loss',
            'behavioral_cloning_loss',
        ]

    def learn_aux(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Overview:
            The auxiliary phase training, where the value is distilled into the policy network
        Returns:
            - aux_loss (:obj:`Tuple[torch.Tensor, torch.Tensor, torch.Tensor]`): including average auxiliary loss\
                average behavioral cloning loss, and average auxiliary value loss
        """
        aux_memories = self._aux_memories
        # gather states and target values into one tensor
        data = {}
        states = []
        actions = []
        return_ = []
        old_values = []
        weights = []
        for memory in aux_memories:
            # for memory in memories:
            states.append(memory['obs'])
            actions.append(memory['action'])
            return_.append(memory['return_'])
            old_values.append(memory['value'])
            if memory['weight'] is None:
                weight = torch.ones_like(memory['action'])
            else:
                weight = torch.tensor(memory['weight'])
            weights.append(weight)

        data['obs'] = torch.cat(states)
        data['action'] = torch.cat(actions)
        data['return_'] = torch.cat(return_)
        data['value'] = torch.cat(old_values)
        data['weight'] = torch.cat(weights)
        # compute current policy logit_old
        with torch.no_grad():
            data['logit_old'] = self._model.forward(data['obs'], mode='compute_actor')['logit']

        # prepared dataloader for auxiliary phase training
        dl = create_shuffled_dataloader(data, self._cfg.learn.batch_size)

        # the proposed auxiliary phase training
        # where the value is distilled into the policy network,
        # while making sure the policy network does not change the action predictions (kl div loss)

        i = 0
        auxiliary_loss_ = 0
        behavioral_cloning_loss_ = 0
        value_loss_ = 0

        for epoch in range(self._aux_train_epoch):
            for data in dl:
                policy_output = self._model.forward(data['obs'], mode='compute_actor_critic')

                # Calculate ppg error 'logit_new', 'logit_old', 'action', 'value_new', 'value_old', 'return_', 'weight'
                data_ppg = ppg_data(
                    policy_output['logit'], data['logit_old'], data['action'], policy_output['value'], data['value'],
                    data['return_'], data['weight']
                )
                ppg_joint_loss = ppg_joint_error(data_ppg, self._clip_ratio)
                wb = self._aux_bc_weight
                total_loss = ppg_joint_loss.auxiliary_loss + wb * ppg_joint_loss.behavioral_cloning_loss

                # # policy network loss copmoses of both the kl div loss as well as the auxiliary loss
                # aux_loss = clipped_value_loss(policy_values, rewards, old_values, self.value_clip)
                # loss_kl = F.kl_div(action_logprobs, old_action_probs, reduction='batchmean')
                # policy_loss = aux_loss + loss_kl

                self._optimizer_ac.zero_grad()
                total_loss.backward()
                self._optimizer_ac.step()

                # paper says it is important to train the value network extra during the auxiliary phase
                # Calculate ppg error 'value_new', 'value_old', 'return_', 'weight'
                values = self._model.forward(data['obs'], mode='compute_critic')['value']
                data_aux = ppo_value_data(values, data['value'], data['return_'], data['weight'])

                value_loss = ppo_value_error(data_aux, self._clip_ratio)

                self._optimizer_aux_critic.zero_grad()
                value_loss.backward()
                self._optimizer_aux_critic.step()

                auxiliary_loss_ += ppg_joint_loss.auxiliary_loss.item()
                behavioral_cloning_loss_ += ppg_joint_loss.behavioral_cloning_loss.item()
                value_loss_ += value_loss.item()
                i += 1

        self._aux_memories = []

        return auxiliary_loss_ / i, behavioral_cloning_loss_ / i, value_loss_ / i
