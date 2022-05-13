from typing import List, Dict, Any, Optional
from collections import deque
import copy
import torch

from ding.utils import list_split, lists_to_dicts
from .gae import gae, gae_data


class Adder(object):
    """
    Overview:
        Adder is a component that handles different transformations and calculations for transitions
        in Collector Module(data generation and processing), such as GAE, n-step return, transition sampling etc.
    Interface:
        __init__, get_gae, get_gae_with_default_last_value, get_nstep_return_data, get_train_sample
    """

    @classmethod
    def get_gae(cls, data: List[Dict[str, Any]], last_value: torch.Tensor, gamma: float, gae_lambda: float,
                cuda: bool) -> List[Dict[str, Any]]:
        """
        Overview:
            Get GAE advantage for stacked transitions(T timestep, 1 batch). Call ``gae`` for calculation.
        Arguments:
            - data (:obj:`list`): Transitions list, each element is a transition dict with at least ['value', 'reward']
            - last_value (:obj:`torch.Tensor`): The last value(i.e.: the T+1 timestep)
            - gamma (:obj:`float`): The future discount factor
            - gae_lambda (:obj:`float`): GAE lambda parameter
            - cuda (:obj:`bool`): Whether use cuda in GAE computation
        Returns:
            - data (:obj:`list`): transitions list like input one, but each element owns extra advantage key 'adv'
        """
        value = torch.stack([d['value'] for d in data])
        next_value = torch.stack([d['value'] for d in data][1:] + [last_value])
        reward = torch.stack([d['reward'] for d in data])
        if cuda:
            value = value.cuda()
            next_value = next_value.cuda()
            reward = reward.cuda()

        adv = gae(gae_data(value, next_value, reward, None, None), gamma, gae_lambda)

        if cuda:
            adv = adv.cpu()
        for i in range(len(data)):
            data[i]['adv'] = adv[i]
        return data

    @classmethod
    def get_gae_with_default_last_value(cls, data: deque, done: bool, gamma: float, gae_lambda: float,
                                        cuda: bool) -> List[Dict[str, Any]]:
        """
        Overview:
            Like ``get_gae`` above to get GAE advantage for stacked transitions. However, this function is designed in
            case ``last_value`` is not passed. If transition is not done yet, it wouold assign last value in ``data``
            as ``last_value``, discard the last element in ``data``(i.e. len(data) would decrease by 1), and then call
            ``get_gae``. Otherwise it would make ``last_value`` equal to 0.
        Arguments:
            - data (:obj:`deque`): Transitions list, each element is a transition dict with \
                at least['value', 'reward']
            - done (:obj:`bool`): Whether the transition reaches the end of an episode(i.e. whether the env is done)
            - gamma (:obj:`float`): The future discount factor
            - gae_lambda (:obj:`float`): GAE lambda parameter
            - cuda (:obj:`bool`): Whether use cuda in GAE computation
        Returns:
            - data (:obj:`List[Dict[str, Any]]`): transitions list like input one, but each element owns \
                extra advantage key 'adv'
        """
        if done:
            last_value = torch.zeros_like(data[-1]['value'])
        else:
            last_data = data.pop()
            last_value = last_data['value']
        return cls.get_gae(data, last_value, gamma, gae_lambda, cuda)

    @classmethod
    def get_nstep_return_data(
            cls,
            data: deque,
            nstep: int,
            cum_reward=False,
            correct_terminate_gamma=True,
            gamma=0.99,
    ) -> deque:
        """
        Overview:
            Process raw traj data by updating keys ['next_obs', 'reward', 'done'] in data's dict element.
        Arguments:
            - data (:obj:`deque`): Transitions list, each element is a transition dict
            - nstep (:obj:`int`): Number of steps. If equals to 1, return ``data`` directly; \
                Otherwise update with nstep value.
        Returns:
            - data (:obj:`deque`): Transitions list like input one, but each element updated with nstep value.
        """
        if nstep == 1:
            return data
        fake_reward = torch.zeros(1)
        next_obs_flag = 'next_obs' in data[0]
        for i in range(len(data) - nstep):
            # update keys ['next_obs', 'reward', 'done'] with their n-step value
            if next_obs_flag:
                data[i]['next_obs'] = data[i + nstep]['obs']  # do not need deepcopy
            if cum_reward:
                data[i]['reward'] = sum([data[i + j]['reward'] * (gamma ** j) for j in range(nstep)])
            else:
                data[i]['reward'] = torch.cat([data[i + j]['reward'] for j in range(nstep)])
            data[i]['done'] = data[i + nstep - 1]['done']
            if correct_terminate_gamma:
                data[i]['value_gamma'] = gamma ** nstep
        for i in range(max(0, len(data) - nstep), len(data)):
            if next_obs_flag:
                data[i]['next_obs'] = data[-1]['next_obs']  # do not need deepcopy
            if cum_reward:
                data[i]['reward'] = sum([data[i + j]['reward'] * (gamma ** j) for j in range(len(data) - i)])
            else:
                data[i]['reward'] = torch.cat(
                    [data[i + j]['reward']
                     for j in range(len(data) - i)] + [fake_reward for _ in range(nstep - (len(data) - i))]
                )
            data[i]['done'] = data[-1]['done']
            if correct_terminate_gamma:
                data[i]['value_gamma'] = gamma ** (len(data) - i - 1)
        return data

    @classmethod
    def get_train_sample(
            cls,
            data: List[Dict[str, Any]],
            unroll_len: int,
            last_fn_type: str = 'last',
            null_transition: Optional[dict] = None
    ) -> List[Dict[str, Any]]:
        """
        Overview:
            Process raw traj data by updating keys ['next_obs', 'reward', 'done'] in data's dict element.
            If ``unroll_len`` equals to 1, which means no process is needed, can directly return ``data``.
            Otherwise, ``data`` will be splitted according to ``unroll_len``, process residual part according to
            ``last_fn_type`` and call ``lists_to_dicts`` to form sampled training data.
        Arguments:
            - data (:obj:`List[Dict[str, Any]]`): Transitions list, each element is a transition dict
            - unroll_len (:obj:`int`): Learn training unroll length
            - last_fn_type (:obj:`str`): The method type name for dealing with last residual data in a traj \
                after splitting, should be in ['last', 'drop', 'null_padding']
            - null_transition (:obj:`Optional[dict]`): Dict type null transition, used in ``null_padding``
        Returns:
            - data (:obj:`List[Dict[str, Any]]`): Transitions list processed after unrolling
        """
        if unroll_len == 1:
            return data
        else:
            # cut data into pieces whose length is unroll_len
            split_data, residual = list_split(data, step=unroll_len)

            def null_padding():
                template = copy.deepcopy(residual[0])
                template['null'] = True
                if isinstance(template['obs'], dict):
                    template['obs'] = {k: torch.zeros_like(v) for k, v in template['obs'].items()}
                else:
                    template['obs'] = torch.zeros_like(template['obs'])
                if 'action' in template:
                    template['action'] = torch.zeros_like(template['action'])
                template['done'] = True
                template['reward'] = torch.zeros_like(template['reward'])
                if 'value_gamma' in template:
                    template['value_gamma'] = 0.
                null_data = [cls._get_null_transition(template, null_transition) for _ in range(miss_num)]
                return null_data

            if residual is not None:
                miss_num = unroll_len - len(residual)
                if last_fn_type == 'drop':
                    # drop the residual part
                    pass
                elif last_fn_type == 'last':
                    if len(split_data) > 0:
                        # copy last datas from split_data's last element, and insert in front of residual
                        last_data = copy.deepcopy(split_data[-1][-miss_num:])
                        split_data.append(last_data + residual)
                    else:
                        # get null transitions using ``null_padding``, and insert behind residual
                        null_data = null_padding()
                        split_data.append(residual + null_data)
                elif last_fn_type == 'null_padding':
                    # same to the case of 'last' type and split_data is empty
                    null_data = null_padding()
                    split_data.append(residual + null_data)
            # collate unroll_len dicts according to keys
            if len(split_data) > 0:
                split_data = [lists_to_dicts(d, recursive=True) for d in split_data]
            return split_data

    @classmethod
    def _get_null_transition(cls, template: dict, null_transition: Optional[dict] = None) -> dict:
        """
        Overview:
            Get null transition for padding. If ``cls._null_transition`` is None, return input ``template`` instead.
        Arguments:
            - template (:obj:`dict`): The template for null transition.
            - null_transition (:obj:`Optional[dict]`): Dict type null transition, used in ``null_padding``
        Returns:
            - null_transition (:obj:`dict`): The deepcopied null transition.
        """
        if null_transition is not None:
            return copy.deepcopy(null_transition)
        else:
            return copy.deepcopy(template)


get_gae = Adder.get_gae
get_gae_with_default_last_value = Adder.get_gae_with_default_last_value
get_nstep_return_data = Adder.get_nstep_return_data
get_train_sample = Adder.get_train_sample
