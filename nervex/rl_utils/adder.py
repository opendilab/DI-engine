from typing import List, Dict, Any, Optional, Callable, Tuple, Union
from collections import deque
import copy
import numpy as np
import torch

from nervex.utils import list_split, lists_to_dicts
from .gae import gae, gae_data


class Adder(object):
    """
    Overview:
        Adder is a component that handles different transformations and calculations for transitions
        in Collector Module(data generation and processing), such as GAE, n-step return, transition sampling etc.
    Interface:
        __init__, get_traj, get_gae, get_gae_with_default_last_value, get_nstep_return_data, get_train_sample, get_her
    """

    def __init__(
            self,
            use_cuda: bool,
            unroll_len: int,
            last_fn_type: str = 'last',
            null_transition: Optional[dict] = None,
            her_strategy: str = 'future',
            her_replay_k: int = 1,
    ) -> None:
        """
        Overview:
            Initialization method for an adder instance
        Arguments:
            - use_cuda (:obj:`bool`): whether use cuda in all the operations
            - unroll_len (:obj:`int`): learn training unroll length
            - last_fn_type (:obj:`str`): the method type name for dealing with last residual data in a traj \
                after splitting, should be in ['last', 'drop', 'null_padding']
            - null_transition (:obj:`Optional[dict]`): dict type null transition, used in ``null_padding``
            - her_strategu (:obj:`str`): the kind of strategy her use, should be in ['final', 'future', 'episode']
            - her_replay_k (:obj:`int`): the num of new timestep generated for a single timestep
        """
        self._use_cuda = use_cuda
        self._device = 'cuda' if self._use_cuda else 'cpu'
        self._unroll_len = unroll_len
        self._last_fn_type = last_fn_type
        assert self._last_fn_type in ['last', 'drop', 'null_padding']
        self._null_transition = null_transition
        self._her_strategy = her_strategy
        assert self._her_strategy in ['final', 'future', 'episode']
        self._her_replay_k = her_replay_k

    def _get_null_transition(self, template: dict) -> dict:
        """
        Overview:
            Get null transition for padding. If ``self._null_transition`` is None, return input ``template`` instead.
        Arguments:
            - template (:obj:`dict`): the template for null transition.
        Returns:
            - null_transition (:obj:`dict`): the deepcopied null transition.
        """
        if self._null_transition is not None:
            return copy.deepcopy(self._null_transition)
        else:
            return copy.deepcopy(template)

    def get_gae(self, data: List[Dict[str, Any]], last_value: torch.Tensor, gamma: float,
                gae_lambda: float) -> List[Dict[str, Any]]:
        """
        Overview:
            Get GAE advantage for stacked transitions(T timestep, 1 batch). Call ``gae`` for calculation.
        Arguments:
            - data (:obj:`list`): transitions list, each element is a transition dict with at least ['value', 'reward']
            - last_value (:obj:`torch.Tensor`): the last value(i.e.: the T+1 timestep)
            - gamma (:obj:`float`): the future discount factor
            - gae_lambda (:obj:`float`): gae lambda parameter
        Returns:
            - data (:obj:`list`): transitions list like input one, but each element owns extra advantage key 'adv'
        """
        value = torch.stack([d['value'] for d in data] + [last_value])
        reward = torch.stack([d['reward'] for d in data])
        if self._use_cuda:
            value = value.cuda()
            reward = reward.cuda()
        adv = gae(gae_data(value, reward), gamma, gae_lambda)
        if self._use_cuda:
            adv = adv.cpu()
        for i in range(len(data)):
            data[i]['adv'] = adv[i]
        return data

    def get_gae_with_default_last_value(self, data: deque, done: bool, gamma: float,
                                        gae_lambda: float) -> List[Dict[str, Any]]:
        """
        Overview:
            Like ``get_gae`` above to get GAE advantage for stacked transitions. However, this function is designed in
            case ``last_value`` is not passed. If transition is not done yet, it wouold assign last value in ``data``
            as ``last_value``, discard the last element in ``data``(i.e. len(data) would decrease by 1), and then call
            ``get_gae``. Otherwise it would make ``last_value`` equal to 0.
        Arguments:
            - data (:obj:`deque`): transitions list, each element is a transition dict with \
                at least['value', 'reward']
            - done (:obj:`bool`): whether the transition reaches the end of an episode(i.e. whether the env is done)
            - gamma (:obj:`float`): the future discount factor
            - gae_lambda (:obj:`float`): gae lambda parameter
        Returns:
            - data (:obj:`List[Dict[str, Any]]`): transitions list like input one, but each element owns \
                extra advantage key 'adv'
        """
        if done:
            last_value = torch.zeros(1)
        else:
            last_data = data.pop()
            last_value = last_data['value']
        return self.get_gae(data, last_value, gamma, gae_lambda)

    def get_nstep_return_data(
            self,
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
            - data (:obj:`deque`): transitions list, each element is a transition dict
            - nstep (:obj:`int`): number of steps. If equals to 1, return ``data`` directly; \
                Otherwise update with nstep value
        Returns:
            - data (:obj:`deque`): transitions list like input one, but each element updated with \
                nstep value
        """
        if nstep == 1:
            return data
        fake_reward = torch.zeros(1)
        next_obs_flag = 'next_obs' in data[0]
        for i in range(len(data) - nstep):
            # update keys ['next_obs', 'reward', 'done'] with their n-step value
            if next_obs_flag:
                data[i]['next_obs'] = data[i + nstep]['obs']
            if cum_reward:
                data[i]['reward'] = sum([data[i + j]['reward'] * (gamma ** j) for j in range(nstep)])
            else:
                data[i]['reward'] = torch.cat([data[i + j]['reward'] for j in range(nstep)])
            data[i]['done'] = data[i + nstep - 1]['done']
            if correct_terminate_gamma:
                data[i]['value_gamma'] = gamma ** nstep
        for i in range(max(0, len(data) - nstep), len(data)):
            if next_obs_flag:
                data[i]['next_obs'] = data[-1]['next_obs']
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

    def get_train_sample(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Overview:
            Process raw traj data by updating keys ['next_obs', 'reward', 'done'] in data's dict element.
            If ``self._unroll_len`` equals to 1, which means no process is needed, can directly return ``data``.
            Otherwise, ``data`` will be splitted according to ``self._unroll_len``, process residual part according to
            ``self._last_fn_type`` and call ``lists_to_dicts`` to form sampled training data.
        Arguments:
            - data (:obj:`List[Dict[str, Any]]`): transitions list, each element is a transition dict
        Returns:
            - data (:obj:`List[Dict[str, Any]]`): transitions list processed after unrolling
        """
        if self._unroll_len == 1:
            return data
        else:
            # cut data into pieces whose length is unroll_len
            split_data, residual = list_split(data, step=self._unroll_len)

            def null_padding():
                template = copy.deepcopy(residual[0])
                template['done'] = True
                template['reward'] = torch.zeros_like(template['reward'])
                if 'value_gamma' in template:
                    template['value_gamma'] = 0.
                null_data = [self._get_null_transition(template) for _ in range(miss_num)]
                return null_data

            if residual is not None:
                miss_num = self._unroll_len - len(residual)
                if self._last_fn_type == 'drop':
                    # drop the residual part
                    pass
                elif self._last_fn_type == 'last':
                    if len(split_data) > 0:
                        # copy last datas from split_data's last element, and insert in front of residual
                        last_data = copy.deepcopy(split_data[-1][-miss_num:])
                        split_data.append(last_data + residual)
                    else:
                        # get null transitions using ``null_padding``, and insert behind residual
                        null_data = null_padding()
                        split_data.append(residual + null_data)
                elif self._last_fn_type == 'null_padding':
                    # same to the case of 'last' type and split_data is empty
                    null_data = null_padding()
                    split_data.append(residual + null_data)
            # collate unroll_len dicts according to keys
            if len(split_data) > 0:
                split_data = [lists_to_dicts(d, recursive=True) for d in split_data]
            return split_data

    def get_her(
            self,
            data: List[Dict[str, Any]],
            merge_func: Optional[Callable] = None,
            split_func: Optional[Callable] = None,
            goal_reward_func: Optional[Callable] = None
    ) -> List[Dict[str, Any]]:
        r"""
        Overview:
            Get HER processed transitions from stacked transitions
        Arguments:
            - data (:obj:`List[Dict[str, Any]]`): transitions list, each element is a transition dict
            - merge_func (:obj:`Callable`): the merge function to use, default set to None. If None, \
                then use ``__her_default_merge_func``
            - split_func (:obj:`Callable`): the split function to use, default set to None. If None, \
                then use ``__her_default_split_func``
            - goal_reward_func (:obj:`Callable`): the goal_reward function to use, default set to None. If None, \
                then use ``__her_default_goal_reward_func``
        Returns:
            - new_data (:obj:`List[Dict[str, Any]]`): the processed transitions
        """
        # TODO(nyz) nstep with her
        # TODO(nyz) unroll_len > 1 with her
        if merge_func is None:
            merge_func = Adder.__her_default_merge_func
        if split_func is None:
            split_func = Adder.__her_default_split_func
        if goal_reward_func is None:
            goal_reward_func = Adder.__her_default_goal_reward_func

        new_data = []
        for idx in range(len(data)):
            obs, _, _ = split_func(data[idx]['obs'])
            next_obs, _, achieved_goal = split_func(data[idx]['next_obs'])
            for k in range(self._her_replay_k):
                if self._her_strategy == 'final':
                    p_idx = -1
                elif self._her_strategy == 'episode':
                    p_idx = np.random.randint(0, len(data))
                elif self._her_strategy == 'future':
                    p_idx = np.random.randint(idx, len(data))
                _, _, new_desired_goal = split_func(data[p_idx]['next_obs'])
                timestep = {k: copy.deepcopy(v) for k, v in data[idx].items() if k not in ['obs', 'next_obs', 'reward']}
                timestep['obs'] = merge_func(obs, new_desired_goal)
                timestep['next_obs'] = merge_func(next_obs, new_desired_goal)
                timestep['reward'] = goal_reward_func(achieved_goal, new_desired_goal).to(self._device)
                new_data.append(timestep)

        return new_data

    @staticmethod
    def __her_default_merge_func(x: Any, y: Any) -> Any:
        r"""
        Overview:
            The function to merge obs in HER timestep
        Arguments:
            - x (:obj:`Any`): one of the timestep obs to merge
            - y (:obj:`Any`): another timestep obs to merge
        Returns:
            - ret (:obj:`Any`): the merge obs
        """
        # TODO(nyz) dict/list merge_func
        return torch.cat([x, y], dim=0)

    @staticmethod
    def __her_default_split_func(x: Any) -> Tuple[Any, Any, Any]:
        r"""
        Overview:
            Split the the obs, achieved goal, desired goal
        Arguments:
            - x (:obj:`Any`): the obs to split
        Returns:
            - obs (:obj:`torch.Tensor`): the split obs
            - desired_goal (:obj:`torch.Tensor`): the split desired_goal
            - achieved_goal (:obj:`torch.Tensor`): the achieved_goal
        """
        # TODO(nyz) dict/list split_func
        # achieved_goal = f(obs), default: f == identical function
        obs, desired_goal = torch.chunk(x, 2)
        achieved_goal = obs
        return obs, desired_goal, achieved_goal

    @staticmethod
    def __her_default_goal_reward_func(achieved_goal: torch.Tensor, desired_goal: torch.Tensor) -> torch.Tensor:
        r"""
        Overview:
            Get the corresponding merge reward according to whether the achieved_goal fit the desired_goal
        Arguments:
            - achieved_goal (:obj:`torch.Tensor`): the achieved goal
            - desired_goal (:obj:`torch.Tensor`): the desired_goal
        Returns:
            - goal_reward (:obj:`torch.Tensor`): the goal reward according to \
            whether the achieved_goal fit the disired_goal
        """
        if (achieved_goal == desired_goal).all():
            return torch.FloatTensor([1])
        else:
            return torch.FloatTensor([0])
