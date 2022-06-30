from dataclasses import dataclass
import time
import torch
import numpy as np
from ding.utils import BUFFER_REGISTRY
import itertools
import random
import logging
from typing import Any, List, Optional, Union
from collections import defaultdict
from ding.data.buffer import Buffer, BufferedData
from ding.utils import fastcopy
from ding.rl_utils.mcts.utils import prepare_observation_lst, concat_output, concat_output_value
# cpp mcts
from ding.rl_utils.mcts.ctree import cytree
from ding.rl_utils.mcts.mcts_ctree import MCTS
# python mcts
# import ding.rl_utils.mcts.ptree as tree
# from ding.rl_utils.mcts.mcts_ptree import EfficientZeroMCTS as MCTS
from ding.model.template.efficientzero.efficientzero_base_model import inverse_scalar_transform
from ding.torch_utils.data_helper import to_ndarray


@dataclass
class BufferedData:
    data: Any
    index: str
    meta: dict


@BUFFER_REGISTRY.register('game')
class GameBuffer(Buffer):

    def __init__(self, config=None):
        """Reference : DISTRIBUTED PRIORITIZED EXPERIENCE REPLAY
        Algo. 1 and Algo. 2 in Page-3 of (https://arxiv.org/pdf/1803.00933.pdf
        """
        super().__init__(config.total_transitions)
        self.config = config
        self.batch_size = config.batch_size
        self.keep_ratio = 1

        self.model_index = 0
        self.model_update_interval = 10

        self.buffer = []
        self.priorities = []
        self.game_history_look_up = []

        self._eps_collected = 0
        self.base_idx = 0
        self._alpha = config.priority_prob_alpha
        self.transition_top = config.total_transitions
        self.clear_time = 0

    def sample_train_data(self, batch_size, policy):
        target_weights = policy._target_model.state_dict()
        batch_context = self.prepare_batch_context(batch_size, self.config.priority_prob_beta)
        input_context = self.make_batch(batch_context, self.config.revisit_policy_search_rate, weights=target_weights)
        reward_value_context, policy_re_context, policy_non_re_context, inputs_batch, target_weights = input_context
        if target_weights is not None:
            policy._target_model.to(self.config.device)
            policy._target_model.eval()

        # target reward, value
        batch_value_prefixs, batch_values = self._prepare_reward_value(reward_value_context, policy._target_model)
        # target policy
        batch_policies_re = self._prepare_policy_re(policy_re_context, policy._target_model)
        batch_policies_non_re = self._prepare_policy_non_re(policy_non_re_context)
        if self.config.revisit_policy_search_rate < 1:
            batch_policies = np.concatenate([batch_policies_re, batch_policies_non_re])
        else:
            batch_policies = batch_policies_re
        targets_batch = [batch_value_prefixs, batch_values, batch_policies]
        # a batch contains the inputs and the targets; inputs is prepared in CPU workers
        # train_data = [inputs_batch, targets_batch]
        # TODO(pu):
        train_data = [inputs_batch, targets_batch, self]
        return train_data

    def push_games(self, data: Any, meta):
        # in EfficientZero replay_buffer.py
        # def save_pools(self, pools, gap_step):
        """
        save a list of game histories
        """
        for (data_game, meta_game) in zip(data, meta):
            self.push(data_game, meta_game)

    def push(self, data: Any, meta: Optional[dict] = None):
        """
        Overview:
            Push data and it's meta information in buffer.
            Save a game history block
        Arguments:
            - data (:obj:`Any`): The data which will be pushed into buffer.
                                 i.e. a game history block
            - meta (:obj:`dict`): Meta information, e.g. priority, count, staleness.
                - end_tag: bool
                    True -> the game is finished. (always True)
                - gap_steps: int
                    if the game is not finished, we only save the transitions that can be computed
                - priorities: list
                    the priorities corresponding to the transitions in the game history
        Returns:
            - buffered_data (:obj:`BufferedData`): The pushed data.
        """
        # in EfficientZero replay_buffer.py
        # def save_game(self, game, end_tag, gap_steps, priorities=None):

        # TODO(pu)
        # if self.get_num_of_transitions() >= self.config.total_transitions:
        #     return

        if meta['end_tag']:
            self._eps_collected += 1
            valid_len = len(data)
        else:
            valid_len = len(data) - meta['gap_steps']

        if meta['priorities'] is None:
            max_prio = self.priorities.max() if self.buffer else 1
            # if no 'priorities' provided, set the valid part of the new-added game history the max_prio
            self.priorities = np.concatenate(
                (self.priorities, [max_prio for _ in range(valid_len)] + [0. for _ in range(valid_len, len(data))])
            )
        else:
            assert len(data) == len(meta['priorities']), " priorities should be of same length as the game steps"
            priorities = meta['priorities'].copy().reshape(-1)
            # priorities[valid_len:len(game)] = 0.
            self.priorities = np.concatenate((self.priorities, priorities))

        self.buffer.append(data)
        self.game_history_look_up += [(self.base_idx + len(self.buffer) - 1, step_pos) for step_pos in range(len(data))]

    def sample(
            self,
            size: Optional[int] = None,
            indices: Optional[List[str]] = None,
            replace: bool = False,
            sample_range: Optional[slice] = None,
            ignore_insufficient: bool = False,
            groupby: str = None,
            rolling_window: int = None
    ) -> Union[List[BufferedData], List[List[BufferedData]]]:
        """
        Overview:
            Sample data with length ``size``.
        Arguments:
            - size (:obj:`Optional[int]`): The number of the data that will be sampled.
            - indices (:obj:`Optional[List[str]]`): Sample with multiple indices.
            - replace (:obj:`bool`): If use replace is true, you may receive duplicated data from the buffer.
            - sample_range (:obj:`slice`): Sample range slice.
            - ignore_insufficient (:obj:`bool`): If ignore_insufficient is true, sampling more than buffer size
                with no repetition will not cause an exception.
            - groupby (:obj:`str`): Groupby key in meta.
            - rolling_window (:obj:`int`): Return batches of window size.
        Returns:
            - sample_data (:obj:`Union[List[BufferedData], List[List[BufferedData]]]`):
                A list of data with length ``size``, may be nested if groupby or rolling_window is set.
        """
        # if size:
        #     if replace:
        #         sampled_indices = np.random.randint(0, self.get_num_of_game_historys(), size)
        #         return [self.buffer[game_history_idx] for game_history_idx in sampled_indices]
        # elif indices:
        #     return [self.buffer[game_history_idx] for game_history_idx in indices]

        storage = self.buffer
        if sample_range:
            storage = list(itertools.islice(self.storage, sample_range.start, sample_range.stop, sample_range.step))

        # Size and indices
        assert size or indices, "One of size and indices must not be empty."
        if (size and indices) and (size != len(indices)):
            raise AssertionError("Size and indices length must be equal.")
        if not size:
            size = len(indices)
        # Indices and groupby
        assert not (indices and groupby), "Cannot use groupby and indicex at the same time."
        # Groupby and rolling_window
        assert not (groupby and rolling_window), "Cannot use groupby and rolling_window at the same time."
        assert not (indices and rolling_window), "Cannot use indices and rolling_window at the same time."

        value_error = None
        sampled_data = []
        if indices:
            # indices_set = set(indices)
            # hashed_data = filter(lambda item: item.index in indices_set, storage)
            # hashed_data = map(lambda item: (item.index, item), hashed_data)
            # hashed_data = dict(hashed_data)
            # # Re-sample and return in indices order
            # sampled_data = [hashed_data[index] for index in indices]
            sampled_data = [self.buffer[game_history_idx] for game_history_idx in indices]

        elif groupby:
            sampled_data = self._sample_by_group(size=size, groupby=groupby, replace=replace, storage=storage)
        elif rolling_window:
            sampled_data = self._sample_by_rolling_window(
                size=size, replace=replace, rolling_window=rolling_window, storage=storage
            )
        else:
            if replace:
                sampled_data = random.choices(storage, k=size)
            else:
                try:
                    sampled_data = random.sample(storage, k=size)
                except ValueError as e:
                    value_error = e

        if value_error or len(sampled_data) != size:
            if ignore_insufficient:
                logging.warning(
                    "Sample operation is ignored due to data insufficient, current buffer is {} while sample is {}".
                    format(self.count(), size)
                )
            else:
                raise ValueError("There are less than {} records/groups in buffer({})".format(size, self.count()))

        return sampled_data

    def get_game(self, idx):
        """
        Overview:
            idx: transition index
            return the game history including this transition
            game_history_idx is the index of this game history in the self.buffer list
            game_history_pos is the relative position of this transition in this game history
        """

        game_history_idx, game_history_pos = self.game_history_look_up[idx]
        game_history_idx -= self.base_idx
        game = self.buffer[game_history_idx]
        return game

    def sample_one_transition(self, idx):
        game_history_idx, game_history_pos = self.game_history_look_up[idx]
        game_history_idx -= self.base_idx
        transition = self.buffer[game_history_idx][game_history_pos]
        return transition

    def get(self, idx: int) -> BufferedData:
        """
        Overview:
            Get item by subscript index
        Arguments:
            - idx (:obj:`int`): Subscript index.  Index of one transition to get.
        Returns:
            - buffered_data (:obj:`BufferedData`): Item from buffer
        """
        game_history_idx, game_history_pos = self.game_history_look_up[idx]
        game_history_idx -= self.base_idx
        game = self.buffer[game_history_idx]
        return game

    def update(self, index, data: Optional[Any] = None, meta: Optional[dict] = None) -> bool:
        """
        Overview:
            Update data and meta by index
        Arguments:
            - index (:obj:`str`): Index of one transition to be updated.
            - data (:obj:`any`): Pure data.  one transition.
            - meta (:obj:`dict`): Meta information.
        Returns:
            - success (:obj:`bool`): Success or not, if data with the index not exist in buffer, return false.
        """

        # update the priorities for data still in replay buffer
        success = False
        # if meta['make_time'] > self.clear_time:
        if index < self.get_num_of_transitions():
            prio = meta['priorities']
            self.priorities[index] = prio
            game_history_idx, game_history_pos = self.game_history_look_up[index]
            game_history_idx -= self.base_idx
            # update one transition
            self.buffer[game_history_idx][game_history_pos] = data
            success = True

        return success

    def batch_update(
            self,
            indices: List[str],
            datas: Optional[List[Optional[Any]]] = None,
            metas: Optional[List[Optional[dict]]] = None
    ) -> None:
        """
        Overview:
            Batch update data and meta by indices, maybe useful in some data architectures.
        Arguments:
            - indices (:obj:`List[str]`): Index of data.
            - datas (:obj:`Optional[List[Optional[Any]]]`): Pure data.
            - metas (:obj:`Optional[List[Optional[dict]]]`): Meta information.
        """
        # def update_priorities(self, batch_indices, batch_priorities, make_time):
        # only update the priorities for data still in replay buffer
        for i in range(len(indices)):
            if metas['make_time'][i] > self.clear_time:
                idx, prio = indices[i], metas['batch_priorities'][i]
                self.priorities[idx] = prio

    def remove_to_fit(self):
        # remove some old data if the replay buffer is full.
        nums_of_game_histoty = self.get_num_of_game_historys()
        total_transition = self.get_num_of_transitions()
        if total_transition > self.transition_top:
            index = 0
            for i in range(nums_of_game_histoty):
                total_transition -= len(self.buffer[i])
                if total_transition <= self.transition_top * self.keep_ratio:
                    index = i
                    break

            if total_transition >= self.config.batch_size:
                self._remove(index + 1)

    def _remove(self, num_excess_games):
        # delete game histories
        excess_games_steps = sum([len(game) for game in self.buffer[:num_excess_games]])
        del self.buffer[:num_excess_games]
        self.priorities = self.priorities[excess_games_steps:]
        del self.game_history_look_up[:excess_games_steps]
        self.base_idx += num_excess_games

        self.clear_time = time.time()

    def delete(self, index: str):
        """
        Overview:
            Delete one data sample by index
        Arguments:
            - index (:obj:`str`): Index
        """
        pass

    def clear(self) -> None:
        del self.buffer[:]

    def get_batch_size(self):
        return self.batch_size

    def get_priorities(self):
        return self.priorities

    def get_num_of_episodes(self):
        # number of collected episodes
        return self._eps_collected

    def get_num_of_game_historys(self) -> int:
        # number of games, i.e. num of game history blocks
        return len(self.buffer)

    def count(self):
        # number of games, i.e. num of game history blocks
        return len(self.buffer)

    def get_num_of_transitions(self):
        # total number of transitions
        return len(self.priorities)

    def __copy__(self) -> "GameBuffer":
        buffer = type(self)(config=self.config)
        buffer.storage = self.buffer
        return buffer

    def prepare_batch_context(self, batch_size, beta):
        """Prepare a batch context that contains:
        game_lst:               a list of game histories
        game_history_pos_lst:           transition index in game (relative index)
        indices_lst:            transition index in replay buffer
        weights_lst:            the weight concerning the priority
        make_time:              the time the batch is made (for correctly updating replay buffer when data is deleted)
        Parameters
        ----------
        batch_size: int
            batch size
        beta: float
            the parameter in PER for calculating the priority
        """
        assert beta > 0

        # total number of transitions
        total = self.get_num_of_transitions()

        probs = self.priorities ** self._alpha

        probs /= probs.sum()
        # TODO sample data in PER way
        # sample according to transition index
        indices_lst = np.random.choice(total, batch_size, p=probs, replace=False)

        weights_lst = (total * probs[indices_lst]) ** (-beta)
        weights_lst /= weights_lst.max()

        game_lst = []
        game_history_pos_lst = []

        for idx in indices_lst:
            game_history_idx, game_history_pos = self.game_history_look_up[idx]
            game_history_idx -= self.base_idx
            game = self.buffer[game_history_idx]

            game_lst.append(game)
            game_history_pos_lst.append(game_history_pos)

        make_time = [time.time() for _ in range(len(indices_lst))]

        context = (game_lst, game_history_pos_lst, indices_lst, weights_lst, make_time)
        return context

    def _prepare_reward_value_context(self, indices, games, state_index_lst, total_transitions):
        """
        prepare the context of rewards and values for reanalyzing part
        Parameters
        ----------
        indices: list
            transition index in replay buffer
        games: list
            list of game histories
        state_index_lst: list
            transition index in game
        total_transitions: int
            number of collected transitions
        """
        zero_obs = games[0].zero_obs()
        config = self.config
        value_obs_lst = []
        # the value is valid or not (out of trajectory)
        value_mask = []
        rewards_lst = []
        traj_lens = []
        # for two_player board games
        action_mask_history, to_play_history = [], []

        td_steps_lst = []
        for game, state_index, idx in zip(games, state_index_lst, indices):
            traj_len = len(game)
            traj_lens.append(traj_len)

            # off-policy correction: shorter horizon of td steps
            delta_td = (total_transitions - idx) // config.auto_td_steps

            td_steps = config.td_steps - delta_td
            td_steps = np.clip(td_steps, 1, 5).astype(np.int)

            # prepare the corresponding observations for bootstrapped values o_{t+k}
            # o[t+ td_steps, t + td_steps + stack frames + num_unroll_steps]
            # t=2+3 -> o[2+3, 2+3+4+5] -> o[5, 14]
            game_obs = game.obs(state_index + td_steps, config.num_unroll_steps)
            rewards_lst.append(game.rewards)

            # for two_player board games
            action_mask_history.append(game.action_mask_history)
            to_play_history.append(game.to_play_history)

            for current_index in range(state_index, state_index + config.num_unroll_steps + 1):
                # get the 6 bootstrapped target obs
                td_steps_lst.append(td_steps)
                # index of bootstrapped obs o_{t+td_steps}
                bootstrap_index = current_index + td_steps

                if bootstrap_index < traj_len:
                    value_mask.append(1)
                    beg_index = bootstrap_index - (state_index + td_steps)
                    end_index = beg_index + config.stacked_observations
                    obs = game_obs[beg_index:end_index]
                else:
                    value_mask.append(0)
                    obs = zero_obs

                value_obs_lst.append(obs)

        reward_value_context = [
            value_obs_lst, value_mask, state_index_lst, rewards_lst, traj_lens, td_steps_lst, action_mask_history,
            to_play_history
        ]
        return reward_value_context

    def _prepare_policy_non_re_context(self, indices, games, state_index_lst):
        """prepare the context of policies for non-reanalyzing part, just return the policy in self-play
        Parameters
        ----------
        indices: list
            transition index in replay buffer
        games: list
            list of game histories
        state_index_lst: list
            transition index in game
        """
        child_visits = []
        traj_lens = []
        # for two_player board games
        action_mask_history, to_play_history = [], []

        for game, state_index, idx in zip(games, state_index_lst, indices):
            traj_len = len(game)
            traj_lens.append(traj_len)
            # for two_player board games
            action_mask_history.append(game.action_mask_history)
            to_play_history.append(game.to_play_history)

            child_visits.append(game.child_visits)

        policy_non_re_context = [state_index_lst, child_visits, traj_lens, action_mask_history, to_play_history]
        return policy_non_re_context

    def _prepare_policy_re_context(self, indices, games, state_index_lst):
        """
        Overview:
            prepare the context of policies for reanalyzing part
        Arguments:
            - indices (:obj:'list'):transition index in replay buffer
            - games (:obj:'list'):list of game histories
            - state_index_lst (:obj:'list'): transition index in game
        """
        zero_obs = games[0].zero_obs()
        config = self.config

        with torch.no_grad():
            # for policy
            policy_obs_lst = []
            policy_mask = []  # 0 -> out of traj, 1 -> new policy
            rewards, child_visits, traj_lens = [], [], []
            # for two_player board games
            action_mask_history, to_play_history = [], []
            for game, state_index in zip(games, state_index_lst):
                traj_len = len(game)
                traj_lens.append(traj_len)
                rewards.append(game.rewards)
                # for two_player board games
                action_mask_history.append(game.action_mask_history)
                to_play_history.append(game.to_play_history)

                child_visits.append(game.child_visits)
                # prepare the corresponding observations
                game_obs = game.obs(state_index, config.num_unroll_steps)
                for current_index in range(state_index, state_index + config.num_unroll_steps + 1):

                    if current_index < traj_len:
                        policy_mask.append(1)
                        beg_index = current_index - state_index
                        end_index = beg_index + config.stacked_observations
                        obs = game_obs[beg_index:end_index]
                    else:
                        policy_mask.append(0)
                        obs = zero_obs
                    policy_obs_lst.append(obs)

        policy_re_context = [
            policy_obs_lst, policy_mask, state_index_lst, indices, child_visits, traj_lens, action_mask_history,
            to_play_history
        ]
        return policy_re_context

    def make_batch(self, batch_context, ratio, weights=None):
        """
        Overview:
            prepare the context of a batch
            reward_value_context:        the context of reanalyzed value targets
            policy_re_context:           the context of reanalyzed policy targets
            policy_non_re_context:       the context of non-reanalyzed policy targets
            inputs_batch:                the inputs of batch
            weights:                     the target model weights
        Arguments:
            batch_context: Any batch context from replay buffer
            ratio: float ratio of reanalyzed policy (value is 100% reanalyzed)
            weights: Any the target model weights
        """
        # obtain the batch context from replay buffer
        game_lst, game_history_pos_lst, indices_lst, weights_lst, make_time_lst = batch_context
        batch_size = len(indices_lst)
        obs_lst, action_lst, mask_lst = [], [], []
        # prepare the inputs of a batch
        for i in range(batch_size):
            game = game_lst[i]
            game_history_pos = game_history_pos_lst[i]

            _actions = game.actions[game_history_pos:game_history_pos + self.config.num_unroll_steps].tolist()
            # add mask for invalid actions (out of trajectory)
            _mask = [1. for i in range(len(_actions))]
            _mask += [0. for _ in range(self.config.num_unroll_steps - len(_mask))]

            _actions += [
                np.random.randint(0, game.action_space_size)
                for _ in range(self.config.num_unroll_steps - len(_actions))
            ]

            # obtain the input observations
            # stack+num_unroll_steps  4+5
            # pad if length of obs in game_history is less than stack+num_unroll_steps
            obs_lst.append(
                game_lst[i].obs(game_history_pos_lst[i], extra_len=self.config.num_unroll_steps, padding=True)
            )
            action_lst.append(_actions)
            mask_lst.append(_mask)

        re_num = int(batch_size * ratio)
        # formalize the input observations
        obs_lst = prepare_observation_lst(obs_lst)

        # formalize the inputs of a batch
        inputs_batch = [obs_lst, action_lst, mask_lst, indices_lst, weights_lst, make_time_lst]
        for i in range(len(inputs_batch)):
            inputs_batch[i] = np.asarray(inputs_batch[i])

        total_transitions = self.get_num_of_transitions()

        # obtain the context of value targets
        reward_value_context = self._prepare_reward_value_context(
            indices_lst, game_lst, game_history_pos_lst, total_transitions
        )

        # 0:re_num -> reanalyzed policy, re_num:end -> non reanalyzed policy
        # reanalyzed policy
        if re_num > 0:
            # obtain the context of reanalyzed policy targets
            policy_re_context = self._prepare_policy_re_context(
                indices_lst[:re_num], game_lst[:re_num], game_history_pos_lst[:re_num]
            )
        else:
            policy_re_context = None

        # non reanalyzed policy
        if re_num < batch_size:
            # obtain the context of non-reanalyzed policy targets
            policy_non_re_context = self._prepare_policy_non_re_context(
                indices_lst[re_num:], game_lst[re_num:], game_history_pos_lst[re_num:]
            )
        else:
            policy_non_re_context = None

        context = reward_value_context, policy_re_context, policy_non_re_context, inputs_batch, weights
        return context

    def _prepare_reward_value(self, reward_value_context, model):
        """
        prepare reward and value targets from the context of rewards and values
        """
        self.model = model
        value_obs_lst, value_mask, state_index_lst, rewards_lst, traj_lens, td_steps_lst, action_mask_history, to_play_history = reward_value_context
        device = self.config.device
        batch_size = len(value_obs_lst)

        if to_play_history[0][0] is not None:
            # for two_player board games
            # to_play
            to_play = []
            for bs in range(batch_size // 6):
                to_play_tmp = list(to_play_history[bs][state_index_lst[bs]:state_index_lst[bs] + 5])
                if len(to_play_tmp) < 6:
                    to_play_tmp += [1 for i in range(6 - len(to_play_tmp))]
                to_play.append(to_play_tmp)
            # to_play = to_ndarray(to_play)
            tmp = []
            for i in to_play:
                tmp += list(i)
            to_play = tmp
            # action_mask
            action_mask = []
            for bs in range(batch_size // 6):
                action_mask_tmp = list(action_mask_history[bs][state_index_lst[bs]:state_index_lst[bs] + 5])
                if len(action_mask_tmp) < 6:
                    action_mask_tmp += [
                        list(np.ones(self.config.action_space_size, dtype=np.int8))
                        for i in range(6 - len(action_mask_tmp))
                    ]
                action_mask.append(action_mask_tmp)
            action_mask = to_ndarray(action_mask)
            tmp = []
            for i in action_mask:
                tmp += i
            action_mask = tmp

        batch_values, batch_value_prefixs = [], []
        with torch.no_grad():
            value_obs_lst = prepare_observation_lst(value_obs_lst)
            # split a full batch into slices of mini_infer_size: to save the GPU memory for more GPU actors
            m_batch = self.config.mini_infer_size
            slices = np.ceil(batch_size / m_batch).astype(np.int_)
            network_output = []
            for i in range(slices):
                beg_index = m_batch * i
                end_index = m_batch * (i + 1)
                if self.config.image_based:
                    m_obs = torch.from_numpy(value_obs_lst[beg_index:end_index]).to(device).float() / 255.0
                else:
                    m_obs = torch.from_numpy(value_obs_lst[beg_index:end_index]).to(device).float()
                # if self.config.amp_type == 'torch_amp':
                #     with autocast():
                #         m_output = self.model.initial_inference(m_obs)
                # else:
                m_output = self.model.initial_inference(m_obs)
                # TODO(pu)
                if not self.model.training:
                    # if not in training, obtain the scalars of the value/reward
                    m_output.hidden_state = m_output.hidden_state.detach().cpu().numpy()
                    m_output.value = inverse_scalar_transform(m_output.value,
                                                              self.config.support_size).detach().cpu().numpy()
                    m_output.policy_logits = m_output.policy_logits.detach().cpu().numpy()
                    m_output.reward_hidden = (
                        m_output.reward_hidden[0].detach().cpu().numpy(),
                        m_output.reward_hidden[1].detach().cpu().numpy()
                    )

                network_output.append(m_output)

            # concat the output slices after model inference
            if self.config.use_root_value:
                # use the root values from MCTS
                # the root values have limited improvement but require much more GPU actors;
                _, value_prefix_pool, policy_logits_pool, hidden_state_roots, reward_hidden_roots = concat_output(
                    network_output
                )
                value_prefix_pool = value_prefix_pool.squeeze().tolist()
                policy_logits_pool = policy_logits_pool.tolist()

                """
                cpp mcts
                """
                roots = cytree.Roots(batch_size, self.config.action_space_size, self.config.num_simulations)
                noises = [
                    np.random.dirichlet([self.config.root_dirichlet_alpha] * self.config.action_space_size
                                        ).astype(np.float32).tolist() for _ in range(batch_size)
                ]
                roots.prepare(
                    self.config.root_exploration_fraction,
                    noises,
                    value_prefix_pool,
                    policy_logits_pool,
                )
                # do MCTS for a new policy with the recent target model
                MCTS(self.config).search(
                    roots, self.model, hidden_state_roots, reward_hidden_roots
                )

                """
                python mcts
                """
                # if to_play_history[0][0] is None:
                #     # for one_player atari games
                #     action_mask = [
                #         list(np.ones(self.config.action_space_size, dtype=np.int8)) for i in range(batch_size)
                #     ]
                # legal_actions = [
                #     [i for i, x in enumerate(action_mask[j]) if x == 1] for j in range(self.config.evaluator_env_num)
                # ]
                # roots = tree.Roots(batch_size, legal_actions, self.config.num_simulations)
                # noises = [
                #     np.random.dirichlet([self.config.root_dirichlet_alpha] * int(sum(action_mask[j]))
                #                         ).astype(np.float32).tolist() for j in range(batch_size)
                # ]
                #
                # if to_play_history[0][0] is None:
                #     roots.prepare(
                #         self.config.root_exploration_fraction,
                #         noises,
                #         value_prefix_pool,
                #         policy_logits_pool,
                #         to_play=None
                #     )
                #     # do MCTS for a new policy with the recent target model
                #     MCTS(self.config).search(roots, self.model, hidden_state_roots, reward_hidden_roots, to_play=None)
                # else:
                #     roots.prepare(
                #         self.config.root_exploration_fraction,
                #         noises,
                #         value_prefix_pool,
                #         policy_logits_pool,
                #         to_play=to_play
                #     )
                #     # do MCTS for a new policy with the recent target model
                #     MCTS(self.config).search(
                #         roots, self.model, hidden_state_roots, reward_hidden_roots, to_play=to_play
                #     )

                roots_values = roots.get_values()
                value_lst = np.array(roots_values)
            else:
                # use the predicted values
                value_lst = concat_output_value(network_output)

            # get last state value
            value_lst = value_lst.reshape(-1) * (
                np.array([self.config.discount for _ in range(batch_size)]) ** td_steps_lst
            )
            value_lst = value_lst * np.array(value_mask)
            value_lst = value_lst.tolist()

            horizon_id, value_index = 0, 0
            for traj_len_non_re, reward_lst, state_index in zip(traj_lens, rewards_lst, state_index_lst):
                # traj_len = len(game)
                target_values = []
                target_value_prefixs = []

                value_prefix = 0.0
                base_index = state_index
                for current_index in range(state_index, state_index + self.config.num_unroll_steps + 1):
                    bootstrap_index = current_index + td_steps_lst[value_index]
                    # for i, reward in enumerate(game.rewards[current_index:bootstrap_index]):
                    for i, reward in enumerate(reward_lst[current_index:bootstrap_index]):
                        value_lst[value_index] += reward * self.config.discount ** i

                    # reset every lstm_horizon_len
                    if horizon_id % self.config.lstm_horizon_len == 0:
                        value_prefix = 0.0
                        base_index = current_index
                    horizon_id += 1

                    if current_index < traj_len_non_re:
                        target_values.append(value_lst[value_index])
                        # Since the horizon is small and the discount is close to 1.
                        # Compute the reward sum to approximate the value prefix for simplification
                        value_prefix += reward_lst[current_index]  # * config.discount ** (current_index - base_index)
                        target_value_prefixs.append(value_prefix)
                    else:
                        target_values.append(0)
                        target_value_prefixs.append(value_prefix)
                    value_index += 1

                batch_value_prefixs.append(target_value_prefixs)
                batch_values.append(target_values)

        batch_value_prefixs = np.asarray(batch_value_prefixs)
        batch_values = np.asarray(batch_values)
        return batch_value_prefixs, batch_values

    def _prepare_policy_re(self, policy_re_context, model):
        """prepare policy targets from the reanalyzed context of policies
        """
        self.model = model
        batch_policies_re = []
        if policy_re_context is None:
            return batch_policies_re

        # for two_player board games
        policy_obs_lst, policy_mask, state_index_lst, indices, child_visits, traj_lens, action_mask_history, to_play_history = policy_re_context
        batch_size = len(policy_obs_lst)
        device = self.config.device

        if to_play_history[0][0] is not None:
            # for two_player board games
            # to_play
            to_play = []
            for bs in range(batch_size // 6):
                to_play_tmp = list(to_play_history[bs][state_index_lst[bs]:state_index_lst[bs] + 5])
                if len(to_play_tmp) < 6:
                    to_play_tmp += [1 for i in range(6 - len(to_play_tmp))]
                to_play.append(to_play_tmp)
            # to_play = to_ndarray(to_play)
            tmp = []
            for i in to_play:
                tmp += list(i)
            to_play = tmp
            # action_mask
            action_mask = []
            for bs in range(batch_size // 6):
                action_mask_tmp = list(action_mask_history[bs][state_index_lst[bs]:state_index_lst[bs] + 5])
                if len(action_mask_tmp) < 6:
                    action_mask_tmp += [
                        list(np.ones(self.config.action_space_size, dtype=np.int8))
                        for i in range(6 - len(action_mask_tmp))
                    ]
                action_mask.append(action_mask_tmp)
            action_mask = to_ndarray(action_mask)
            tmp = []
            for i in action_mask:
                tmp += i
            action_mask = tmp

        with torch.no_grad():
            # TODO
            policy_obs_lst = prepare_observation_lst(policy_obs_lst)
            # split a full batch into slices of mini_infer_size: to save the GPU memory for more GPU actors
            m_batch = self.config.mini_infer_size
            slices = np.ceil(batch_size / m_batch).astype(np.int_)
            network_output = []
            for i in range(slices):
                beg_index = m_batch * i
                end_index = m_batch * (i + 1)

                m_obs = torch.from_numpy(policy_obs_lst[beg_index:end_index]).to(device).float() / 255.0
                m_output = self.model.initial_inference(m_obs)
                # TODO(pu)
                if not self.model.training:
                    # if not in training, obtain the scalars of the value/reward
                    m_output.hidden_state = m_output.hidden_state.detach().cpu().numpy()
                    m_output.value = inverse_scalar_transform(m_output.value,
                                                              self.config.support_size).detach().cpu().numpy()
                    m_output.policy_logits = m_output.policy_logits.detach().cpu().numpy()
                    m_output.reward_hidden = (
                        m_output.reward_hidden[0].detach().cpu().numpy(),
                        m_output.reward_hidden[1].detach().cpu().numpy()
                    )
                network_output.append(m_output)

            _, value_prefix_pool, policy_logits_pool, hidden_state_roots, reward_hidden_roots = concat_output(
                network_output
            )
            value_prefix_pool = value_prefix_pool.squeeze().tolist()
            policy_logits_pool = policy_logits_pool.tolist()

            """
            cpp mcts
            """
            roots = cytree.Roots(batch_size, self.config.action_space_size, self.config.num_simulations)
            noises = [
                np.random.dirichlet([self.config.root_dirichlet_alpha] * self.config.action_space_size
                                    ).astype(np.float32).tolist() for _ in range(batch_size)
            ]
            roots.prepare(
                self.config.root_exploration_fraction,
                noises,
                value_prefix_pool,
                policy_logits_pool,
            )
            # do MCTS for a new policy with the recent target model
            MCTS(self.config).search(roots, self.model, hidden_state_roots, reward_hidden_roots)

            """
            python mcts
            """
            # if to_play_history[0][0] is None:
            #     # for one_player atari games
            #     action_mask = [list(np.ones(self.config.action_space_size, dtype=np.int8)) for i in range(batch_size)]
            # legal_actions = [[i for i, x in enumerate(action_mask[j]) if x == 1] for j in range(batch_size)]
            # roots = tree.Roots(batch_size, legal_actions, self.config.num_simulations)
            # noises = [
            #     np.random.dirichlet([self.config.root_dirichlet_alpha] * int(sum(action_mask[j]))
            #                         ).astype(np.float32).tolist() for j in range(batch_size)
            # ]
            # if to_play_history[0][0] is None:
            #     roots.prepare(
            #         self.config.root_exploration_fraction, noises, value_prefix_pool, policy_logits_pool, to_play=None
            #     )
            #     # do MCTS for a new policy with the recent target model
            #     MCTS(self.config).search(roots, self.model, hidden_state_roots, reward_hidden_roots, to_play=None)
            # else:
            #     roots.prepare(
            #         self.config.root_exploration_fraction,
            #         noises,
            #         value_prefix_pool,
            #         policy_logits_pool,
            #         to_play=to_play
            #     )
            #     # do MCTS for a new policy with the recent target model
            #     MCTS(self.config).search(roots, self.model, hidden_state_roots, reward_hidden_roots, to_play=to_play)
            # roots_legal_actions_list = roots.legal_actions_list

            roots_distributions = roots.get_distributions()

            policy_index = 0
            for state_index, game_idx in zip(state_index_lst, indices):
                target_policies = []

                for current_index in range(state_index, state_index + self.config.num_unroll_steps + 1):
                    distributions = roots_distributions[policy_index]

                    if policy_mask[policy_index] == 0:
                        target_policies.append([0 for _ in range(self.config.action_space_size)])
                    else:
                        if distributions is None:
                            # if at obs, the legal_action is None, fake target_policy
                            target_policies.append(
                                list(np.ones(self.config.action_space_size) / self.config.action_space_size)
                            )
                        else:
                            """
                            cpp mcts
                            """
                            # for one_player atari games
                            sum_visits = sum(distributions)
                            policy = [visit_count / sum_visits for visit_count in distributions]
                            target_policies.append(policy)

                            """
                            python mcts
                            """
                            # if to_play_history[0][0] is None:
                            #     # for one_player atari games
                            #     sum_visits = sum(distributions)
                            #     policy = [visit_count / sum_visits for visit_count in distributions]
                            #     target_policies.append(policy)
                            # else:
                            #     # for two_player board games
                            #     policy_tmp = [0 for _ in range(self.config.action_space_size)]
                            #     # to make sure  target_policies have the same dimension
                            #     # target_policy = torch.from_numpy(target_policy) be correct
                            #     sum_visits = sum(distributions)
                            #     policy = [visit_count / sum_visits for visit_count in distributions]
                            #     for index, legal_action in enumerate(roots_legal_actions_list[policy_index]):
                            #         policy_tmp[legal_action] = policy[index]
                            #     target_policies.append(policy_tmp)

                    policy_index += 1

                batch_policies_re.append(target_policies)

        batch_policies_re = np.array(batch_policies_re)

        return batch_policies_re

    def _prepare_policy_non_re(self, policy_non_re_context):
        """prepare policy targets from the non-reanalyzed context of policies
        """
        batch_policies_non_re = []
        if policy_non_re_context is None:
            return batch_policies_non_re

        state_index_lst, child_visits, traj_lens = policy_non_re_context
        with torch.no_grad():
            # for policy
            policy_mask = []  # 0 -> out of traj, 1 -> old policy
            # for game, state_index in zip(games, state_index_lst):
            for traj_len, child_visit, state_index in zip(traj_lens, child_visits, state_index_lst):
                # traj_len = len(game)
                target_policies = []

                for current_index in range(state_index, state_index + self.config.num_unroll_steps + 1):
                    if current_index < traj_len:
                        target_policies.append(child_visit[current_index])
                        policy_mask.append(1)
                    else:
                        target_policies.append([0 for _ in range(self.config.action_space_size)])
                        policy_mask.append(0)

                batch_policies_non_re.append(target_policies)
        batch_policies_non_re = np.asarray(batch_policies_non_re)
        return batch_policies_non_re
