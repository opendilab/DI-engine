from typing import Any, List, Optional, Union, Callable
from dataclasses import dataclass
from functools import wraps
import time

import numpy as np
from typing import Any, Iterable, List, Optional, Tuple, Union

from ding.worker.buffer import Buffer, apply_middleware, BufferedData


def apply_middleware(func_name: str):

    def wrap_func(base_func: Callable):

        @wraps(base_func)
        def handler(buffer, *args, **kwargs):
            """
            Overview:
                The real processing starts here, we apply the middleware one by one,
                each middleware will receive next `chained` function, which is an executor of next
                middleware. You can change the input arguments to the next `chained` middleware, and you
                also can get the return value from the next middleware, so you have the
                maximum freedom to choose at what stage to implement your method.
            """

            def wrap_handler(middleware, *args, **kwargs):
                if len(middleware) == 0:
                    return base_func(buffer, *args, **kwargs)

                def chain(*args, **kwargs):
                    return wrap_handler(middleware[1:], *args, **kwargs)

                func = middleware[0]
                return func(func_name, chain, *args, **kwargs)

            return wrap_handler(buffer.middleware, *args, **kwargs)

        return handler

    return wrap_func


@dataclass
class BufferedData:
    data: Any
    index: str
    meta: dict


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
        # save a list of game histories
        for (data_game, meta_game) in (data, meta):
            # Only append end game
            # if end_tag:
            # self.push(game, True, gap_step, priorities)
            self.push(data_game, meta_game)

    def push(self, data: Any, meta: Optional[dict] = None):
        """
        Overview:
            Push data and it's meta information in buffer.
        Arguments:
            - data (:obj:`Any`): The data which will be pushed into buffer.
            - meta (:obj:`dict`): Meta information, e.g. priority, count, staleness.
        Returns:
            - buffered_data (:obj:`BufferedData`): The pushed data.
        """
        # in EfficientZero replay_buffer.py
        # def save_game(self, game, end_tag, gap_steps, priorities=None):
        #     """Save a game history block
        #     Parameters
        #     ----------
        #     game: Any
        #         a game history block
        #     end_tag: bool
        #         True -> the game is finished. (always True)
        #     gap_steps: int
        #         if the game is not finished, we only save the transitions that can be computed
        #     priorities: list
        #         the priorities corresponding to the transitions in the game history
        #     """

        if self.get_total_len() >= self.config.total_transitions:
            return

        if meta['end_tag']:
            self._eps_collected += 1
            valid_len = len( data)
        else:
            valid_len = len(data) - meta['gap_steps']

        if meta['priorities'] is None:
            max_prio = self.priorities.max() if self.buffer else 1
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
        pass
        # raise NotImplementedError

    def sample_game(self, idx):
        # def get_game() in EfficientZero replay_buffer.py

        # return a game
        game_id, game_pos = self.game_look_up[idx]
        game_id -= self.base_idx
        game = self.buffer[game_id]
        return game

    def prepare_batch_context(self, batch_size, beta):
        """Prepare a batch context that contains:
        game_lst:               a list of game histories
        game_pos_lst:           transition index in game (relative index)
        indices_lst:            transition index in replay buffer
        weights_lst:            the weight concering the priority
        make_time:              the time the batch is made (for correctly updating replay buffer when data is deleted)
        Parameters
        ----------
        batch_size: int
            batch size
        beta: float
            the parameter in PER for calculating the priority
        """
        assert beta > 0

        total = self.get_total_len()

        probs = self.priorities ** self._alpha

        probs /= probs.sum()
        # sample data
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

    def update(self, index: str, data: Optional[Any] = None, meta: Optional[dict] = None) -> bool:
        """
        Overview:
            Update data and meta by index
        Arguments:
            - index (:obj:`str`): Index of data.
            - data (:obj:`any`): Pure data.
            - meta (:obj:`dict`): Meta information.
        Returns:
            - success (:obj:`bool`): Success or not, if data with the index not exist in buffer, return false.
        """
        # raise NotImplementedError

        # def update_priorities(self, batch_indices, batch_priorities, make_time):

        # update the priorities for data still in replay buffer
        if meta['make_time'] > self.clear_time:
            idx, prio = index, meta['batch_priorities']
            self.priorities = prio

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
        # update the priorities for data still in replay buffer
        for i in range(len(indices)):
            if metas[i]['make_time']> self.clear_time:
                idx, prio = indices[i], metas[i]['batch_priorities']
                self.priorities[idx] = prio

    def remove_to_fit(self):
        # remove some old data if the replay buffer is full.
        current_size = self.count()
        total_transition = self.get_total_len()
        if total_transition > self.transition_top:
            index = 0
            for i in range(current_size):
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

    def count(self) -> int:
        # def size(self):
            # number of games
            return len(self.buffer)

    def clear(self) -> None:
        del self.buffer[:]

    def get(self, idx: int) -> BufferedData:
        """
        Overview:
            Get item by subscript index
        Arguments:
            - idx (:obj:`int`): Subscript index
        Returns:
            - buffered_data (:obj:`BufferedData`): Item from buffer
        """
        # def get_game(self, idx):
        # return a game
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

    def get_total_len(self):
        # number of transitions
        return len(self.priorities)

