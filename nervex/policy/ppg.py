from typing import List, Dict, Any, Tuple, Union, Optional
from collections import namedtuple, deque
import copy
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from nervex.utils import POLICY_REGISTRY, squeeze
from nervex.data import default_collate
from nervex.torch_utils import Adam, to_device
from nervex.rl_utils import \
    ppo_policy_data, ppo_policy_error, Adder, ppo_value_data, ppo_value_error, ppg_data, ppg_joint_error
from nervex.model import FCValueAC, ConvValueAC
from nervex.armor import Armor
from .base_policy import Policy
from .common_policy import CommonPolicy


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
class PPGPolicy(CommonPolicy):
    r"""
    Overview:
        Policy class of PPG algorithm.
    """

    def _init_learn(self) -> None:
        r"""
        Overview:
            Learn mode init method. Called by ``self.__init__``.
            Init the optimizer, algorithm config and the main armor.
        """
        # Optimizer
        self._optimizer_policy = Adam(self._model._policy_net.parameters(), lr=self._cfg.learn.learning_rate)
        self._optimizer_value = Adam(self._model._value_net.parameters(), lr=self._cfg.learn.learning_rate)
        self._armor = Armor(self._model)

        # Algorithm config
        algo_cfg = self._cfg.learn.algo
        self._value_weight = algo_cfg.value_weight
        self._entropy_weight = algo_cfg.entropy_weight
        self._clip_ratio = algo_cfg.clip_ratio
        self._use_adv_norm = algo_cfg.get('use_adv_norm', False)

        # Main armor
        self._armor.mode(train=True)
        self._armor.reset()

        # Auxiliary memories
        self._epochs_aux = algo_cfg.epochs_aux
        self._train_iteration = 0
        self._aux_memories = []
        self._beta_weight = algo_cfg.beta_weight

    def _data_preprocess_learn(self, data: List[Any]) -> Tuple[dict, dict]:
        # TODO(nyz) priority for ppg
        use_priority = self._cfg.get('use_priority', False)
        assert not use_priority, "NotImplement"
        data_info = {}
        # data preprocess
        for k, data_item in data.items():
            data_item = default_collate(data_item)
            ignore_done = self._cfg.learn.get('ignore_done', False)
            if ignore_done:
                data_item['done'] = None
            else:
                data_item['done'] = data_item['done'].float()
            data_item['weight'] = None
            data[k] = data_item
        if self._use_cuda:
            data = to_device(data, self._device)
        return data, data_info

    def _forward_learn(self, data: dict) -> Dict[str, Any]:
        r"""
        Overview:
            Forward and backward function of learn mode.
        Arguments:
            - data (:obj:`dict`): Dict type data
        Returns:
            - info_dict (:obj:`Dict[str, Any]`):
              Including current lr, total_loss, policy_loss, value_loss, entropy_loss, \
                        adv_abs_max, approx_kl, clipfrac
        """
        # ====================
        # PPG forward
        # ====================
        policy_data, value_data = data['policy'], data['value']
        policy_adv, value_adv = policy_data['adv'], value_data['adv']
        if self._use_adv_norm:
            # Normalize advantage in a total train_batch
            policy_adv = (policy_adv - policy_adv.mean()) / (policy_adv.std() + 1e-8)
            value_adv = (value_adv - value_adv.mean()) / (value_adv.std() + 1e-8)
        # Policy Phase(Policy)
        policy_output = self._armor.forward(policy_data, param={'mode': 'compute_action'})
        policy_error_data = ppo_policy_data(
            policy_output['logit'], policy_data['logit'], policy_data['action'], policy_adv, policy_data['weight']
        )
        ppo_policy_loss, ppo_info = ppo_policy_error(policy_error_data, self._clip_ratio)
        policy_loss = ppo_policy_loss.policy_loss - self._entropy_weight * ppo_policy_loss.entropy_loss
        self._optimizer_policy.zero_grad()
        policy_loss.backward()
        self._optimizer_policy.step()

        # Policy Phase(Value)
        return_ = value_data['value'] + value_adv
        value_output = self._armor.forward(value_data, param={'mode': 'compute_value'})
        value_error_data = ppo_value_data(value_output['value'], value_data['value'], return_, value_data['weight'])
        value_loss = self._value_weight * ppo_value_error(value_error_data, self._clip_ratio)
        self._optimizer_value.zero_grad()
        value_loss.backward()
        self._optimizer_value.step()

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
        if self._train_iteration % self._cfg.learn.algo.aux_freq == 0:
            aux_loss, bc_loss, aux_value_loss = self.learn_aux()
            return {
                'policy_cur_lr': self._optimizer_policy.defaults['lr'],
                'value_cur_lr': self._optimizer_value.defaults['lr'],
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
                'policy_cur_lr': self._optimizer_policy.defaults['lr'],
                'value_cur_lr': self._optimizer_value.defaults['lr'],
                'policy_loss': ppo_policy_loss.policy_loss.item(),
                'value_loss': value_loss.item(),
                'entropy_loss': ppo_policy_loss.entropy_loss.item(),
                'policy_adv_abs_max': policy_adv.abs().max().item(),
                'approx_kl': ppo_info.approx_kl,
                'clipfrac': ppo_info.clipfrac,
            }

    def _init_collect(self) -> None:
        r"""
        Overview:
            Collect mode init method. Called by ``self.__init__``.
            Init unroll length, adder, collect armor.
        """
        self._unroll_len = self._cfg.collect.unroll_len
        self._collect_armor = Armor(self._model)
        # TODO continuous action space exploration
        self._collect_armor.add_plugin('main', 'multinomial_sample')
        self._collect_armor.mode(train=False)
        self._collect_armor.reset()
        self._adder = Adder(self._use_cuda, self._unroll_len)
        algo_cfg = self._cfg.collect.algo
        self._gamma = algo_cfg.discount_factor
        self._gae_lambda = algo_cfg.gae_lambda

    def _forward_collect(self, data_id: List[int], data: dict) -> dict:
        r"""
        Overview:
            Forward function for collect mode
        Arguments:
            - data_id (:obj:`List` of :obj:`int`): Not used, set in arguments for consistency
            - data (:obj:`dict`): Dict type data, including at least ['obs'].
        Returns:
            - data (:obj:`dict`): The collected data
        """
        with torch.no_grad():
            output = self._collect_armor.forward(data, param={'mode': 'compute_action_value'})
        return output

    def _process_transition(self, obs: Any, armor_output: dict, timestep: namedtuple) -> dict:
        """
        Overview:
               Generate dict type transition data from inputs.
        Arguments:
                - obs (:obj:`Any`): Env observation
                - armor_output (:obj:`dict`): Output of collect armor, including at least ['action']
                - timestep (:obj:`namedtuple`): Output after env step, including at least ['obs', 'reward', 'done']\
                       (here 'obs' indicates obs after env step).
        Returns:
               - transition (:obj:`dict`): Dict type transition data.
        """
        transition = {
            'obs': obs,
            'logit': armor_output['logit'],
            'action': armor_output['action'],
            'value': armor_output['value'],
            'reward': timestep.reward,
            'done': timestep.done,
        }
        return transition

    def _get_train_sample(self, data: deque) -> Union[None, List[Any]]:
        r"""
        Overview:
            Get the trajectory and calculate GAE, return one data to cache for next time calculation
        Arguments:
            - data (:obj:`deque`): The trajectory's cache
        Returns:
            - samples (:obj:`dict`): The training samples generated
        """
        # adder is defined in _init_collect
        data = self._adder.get_gae_with_default_last_value(
            data, data[-1]['done'], gamma=self._gamma, gae_lambda=self._gae_lambda
        )
        data = self._adder.get_train_sample(data)
        for d in data:
            d['buffer_name'] = ["policy", "value"]
        return data

    def _get_batch_size(self) -> Dict[str, int]:
        bs = self._cfg.learn.batch_size
        return {'policy': bs, 'value': bs}

    def _init_eval(self) -> None:
        r"""
        Overview:
            Evaluate mode init method. Called by ``self.__init__``.
            Init eval armor with argmax strategy.
        """
        self._eval_armor = Armor(self._model)
        self._eval_armor.add_plugin('main', 'argmax_sample')
        self._eval_armor.mode(train=False)
        self._eval_armor.reset()

    def _forward_eval(self, data_id: List[int], data: dict) -> dict:
        r"""
        Overview:
            Forward function for eval mode, similar to ``self._forward_collect``.
        Arguments:
            - data_id (:obj:`List[int]`): Not used in this policy.
            - data (:obj:`dict`): Dict type data, including at least ['obs'].
        Returns:
            - output (:obj:`dict`): Dict type data, including at least inferred action according to input obs.
        """
        with torch.no_grad():
            output = self._eval_armor.forward(data, param={'mode': 'compute_action'})
        return output

    def default_model(self) -> Tuple[str, List[str]]:
        # return 'fc_vac', ['nervex.model.actor_critic.value_ac']
        return 'ppg', ['nervex.model.ppg']

    def _monitor_vars_learn(self) -> List[str]:
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
            data['logit_old'] = self._armor.forward({'obs': data['obs']}, param={'mode': 'compute_action'})['logit']

        # prepared dataloader for auxiliary phase training
        dl = create_shuffled_dataloader(data, self._cfg.learn.batch_size)

        # the proposed auxiliary phase training
        # where the value is distilled into the policy network,
        # while making sure the policy network does not change the action predictions (kl div loss)

        i = 0
        auxiliary_loss_ = 0
        behavioral_cloning_loss_ = 0
        value_loss_ = 0

        for epoch in range(self._epochs_aux):
            for data in dl:
                policy_output = self._armor.forward(data, param={'mode': 'compute_action_value'})

                # Calculate ppg error 'logit_new', 'logit_old', 'action', 'value_new', 'value_old', 'return_', 'weight'
                data_ppg = ppg_data(
                    policy_output['logit'], data['logit_old'], data['action'], policy_output['value'], data['value'],
                    data['return_'], data['weight']
                )
                ppg_joint_loss = ppg_joint_error(data_ppg, self._clip_ratio)
                wb = self._beta_weight
                total_loss = ppg_joint_loss.auxiliary_loss + wb * ppg_joint_loss.behavioral_cloning_loss

                # # policy network loss copmoses of both the kl div loss as well as the auxiliary loss
                # aux_loss = clipped_value_loss(policy_values, rewards, old_values, self.value_clip)
                # loss_kl = F.kl_div(action_logprobs, old_action_probs, reduction='batchmean')
                # policy_loss = aux_loss + loss_kl

                self._optimizer_policy.zero_grad()
                total_loss.backward()
                self._optimizer_policy.step()

                # paper says it is important to train the value network extra during the auxiliary phase
                # Calculate ppg error 'value_new', 'value_old', 'return_', 'weight'
                values = self._armor.forward(data, param={'mode': 'compute_value'})['value']
                data_aux = ppo_value_data(values, data['value'], data['return_'], data['weight'])

                value_loss = ppo_value_error(data_aux, self._clip_ratio)

                self._optimizer_value.zero_grad()
                value_loss.backward()
                self._optimizer_value.step()

                auxiliary_loss_ += ppg_joint_loss.auxiliary_loss.item()
                behavioral_cloning_loss_ += ppg_joint_loss.behavioral_cloning_loss.item()
                value_loss_ += value_loss.item()
                i += 1

        self._aux_memories = []

        return auxiliary_loss_ / i, behavioral_cloning_loss_ / i, value_loss_ / i
