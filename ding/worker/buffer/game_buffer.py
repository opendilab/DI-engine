from dataclasses import dataclass
import time
import numpy as np
from ding.utils import BUFFER_REGISTRY
import itertools
import random
import logging
from typing import Any, Iterable, List, Optional, Tuple, Union
from collections import defaultdict, deque, OrderedDict
from ding.worker.buffer import Buffer, apply_middleware, BufferedData
from ding.worker.buffer.utils import fastcopy


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
        super().__init__()
        self.config = config
        self.batch_size = config.batch_size
        self.keep_ratio = 1

        self.model_index = 0
        self.model_update_interval = 10

        self.buffer = []
        self.priorities = []
        self.game_look_up = []

        self._eps_collected = 0
        self.base_idx = 0
        self._alpha = config.priority_prob_alpha
        self.transition_top = int(config.transition_num * 10 ** 6)
        self.clear_time = 0

    def push_games(self, data: Any, meta):
        # in EfficientZero replay_buffer.py
        # def save_pools(self, pools, gap_step):
        """save a list of game histories
        """
        for (data_game, meta_game) in zip(data, meta):
            # Only append end game
            # if end_tag:
            # self.push(game, True, gap_step, priorities)
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

        if self.get_total_num_transitions() >= self.config.total_transitions:
            return

        if meta['end_tag']:
            self._eps_collected += 1
            valid_len = len(data)
        else:
            valid_len = len(data) - meta['gap_steps']

        if meta['priorities'] is None:
            max_prio = self.priorities.max() if self.buffer else 1
            # if no 'priorities' provided, set the valid part of the new-added game history the max_prio
            self.priorities = np.concatenate(
                (self.priorities, [max_prio for _ in range(valid_len)] + [0. for _ in range(valid_len, len(data))]))
        else:
            assert len(data) == len(meta['priorities']), " priorities should be of same length as the game steps"
            priorities = meta['priorities'].copy().reshape(-1)
            # priorities[valid_len:len(game)] = 0.
            self.priorities = np.concatenate((self.priorities, priorities))

        self.buffer.append(data)
        self.game_look_up += [(self.base_idx + len(self.buffer) - 1, step_pos) for step_pos in range(len(data))]

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
        #         return [self.buffer[game_id] for game_id in sampled_indices]
        # elif indices:
        #     return [self.buffer[game_id] for game_id in indices]

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
            sampled_data = [self.buffer[game_id] for game_id in indices]

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

        # sampled_data = self._independence(sampled_data)

        return sampled_data

    def _independence(
        self, buffered_samples: Union[List[BufferedData], List[List[BufferedData]]]
    ) -> Union[List[BufferedData], List[List[BufferedData]]]:
        """
        Overview:
            Make sure that each record is different from each other, but remember that this function
            is different from clone_object. You may change the data in the buffer by modifying a record.
        Arguments:
            - buffered_samples (:obj:`Union[List[BufferedData], List[List[BufferedData]]]`) Sampled data,
                can be nested if groupby or rolling_window has been set.
        """
        if len(buffered_samples) == 0:
            return buffered_samples
        occurred = defaultdict(int)

        for i, buffered in enumerate(buffered_samples):
            if isinstance(buffered, list):
                sampled_list = buffered
                # Loop over nested samples
                for j, buffered in enumerate(sampled_list):
                    occurred[buffered.index] += 1
                    if occurred[buffered.index] > 1:
                        sampled_list[j] = fastcopy.copy(buffered)
            elif isinstance(buffered, BufferedData):
                occurred[buffered.index] += 1
                if occurred[buffered.index] > 1:
                    buffered_samples[i] = fastcopy.copy(buffered)
            else:
                raise Exception("Get unexpected buffered type {}".format(type(buffered)))
        return buffered_samples

    def get_game(self, idx):
        # def get_game() in EfficientZero replay_buffer.py
        # idx: transition index
        # return the game history including this transition

        # game_id is the index of this game history in the self.buffer list
        # game_pos is the relative position of this transition in this game history
        game_id, game_pos = self.game_look_up[idx]
        game_id -= self.base_idx
        game = self.buffer[game_id]
        return game

    def sample_one_transition(self, idx):
        game_id, game_pos = self.game_look_up[idx]
        game_id -= self.base_idx
        transition = self.buffer[game_id][game_pos]
        return transition

    def prepare_batch_context(self, batch_size, beta):
        """Prepare a batch context that contains:
        game_lst:               a list of game histories
        game_pos_lst:           transition index in game (relative index)
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
        total = self.get_total_num_transitions()

        probs = self.priorities ** self._alpha

        probs /= probs.sum()
        # TODO sample data in PER way
        # sample according to transition index
        indices_lst = np.random.choice(total, batch_size, p=probs, replace=False)

        weights_lst = (total * probs[indices_lst]) ** (-beta)
        weights_lst /= weights_lst.max()

        game_lst = []
        game_pos_lst = []

        for idx in indices_lst:
            game_id, game_pos = self.game_look_up[idx]
            game_id -= self.base_idx
            game = self.buffer[game_id]

            game_lst.append(game)
            game_pos_lst.append(game_pos)

        make_time = [time.time() for _ in range(len(indices_lst))]

        context = (game_lst, game_pos_lst, indices_lst, weights_lst, make_time)
        return context

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
        # raise NotImplementedError

        # def update_priorities(self, batch_indices, batch_priorities, make_time):

        # update the priorities for data still in replay buffer
        success = False
        # if meta['make_time'] > self.clear_time:
        if index < self.get_total_num_transitions():
            prio = meta['priorities']
            self.priorities[index] = prio
            game_id, game_pos = self.game_look_up[index]
            game_id -= self.base_idx
            # update one transition
            self.buffer[game_id][game_pos] = data
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
            if metas[i]['make_time'] > self.clear_time:
                idx, prio = indices[i], metas[i]['batch_priorities']
                self.priorities[idx] = prio

    def remove_to_fit(self):
        # remove some old data if the replay buffer is full.
        nums_of_game_histoty = self.get_num_of_game_historys()
        total_transition = self.get_total_num_transitions()
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
        del self.game_look_up[:excess_games_steps]
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

    def get_num_of_game_historys(self) -> int:
        # number of games, i.e. num  of game history blocks
        return len(self.buffer)

    def count(self):
        # number of games, i.e. num  of game history blocks
        return len(self.buffer)

    def clear(self) -> None:
        del self.buffer[:]

    def get(self, idx: int) -> BufferedData:
        """
        Overview:
            Get item by subscript index
        Arguments:
            - idx (:obj:`int`): Subscript index.  Index of one transition to get.
        Returns:
            - buffered_data (:obj:`BufferedData`): Item from buffer
        """
        # def get_game(self, idx):
        # return a game
        #
        game_id, game_pos = self.game_look_up[idx]
        game_id -= self.base_idx
        game = self.buffer[game_id]
        return game

    def episodes_collected(self):
        # number of collected histories
        return self._eps_collected

    def get_batch_size(self):
        return self.batch_size

    def get_priorities(self):
        return self.priorities

    def get_total_num_transitions(self):
        # number of transitions
        return len(self.priorities)

    def __copy__(self) -> "GameBuffer":
        buffer = type(self)(config=self.config)
        buffer.storage = self.buffer
        return buffer
