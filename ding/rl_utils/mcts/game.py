"""
The following code is adapted from https://github.com/YeWR/EfficientZero/blob/main/core/game.py
"""

import copy
import numpy as np
from ding.utils.compression_helper import jpeg_data_decompressor


class Game:

    def __init__(self, env, action_space_size: int, config=None):
        self.env = env
        self.action_space_size = action_space_size
        self.config = config

    def legal_actions(self):
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError()

    def close(self, *args, **kwargs):
        self.env.close(*args, **kwargs)

    def render(self, *args, **kwargs):
        self.env.render(*args, **kwargs)


class GameHistory:
    """
        Overview:
            A block of game history from a full episode trajectories.
            The length of one episode in Atari games are quite large. Split the whole episode trajectory into several
            ``GameHistory`` blocks.
        Interfaces:
            ``__init__``, ``__len__``,``init``, ``pad_over``, ``is_full``, ``legal_actions``, ``append``, ``obs``
            ``zero_obs``, ``step_obs``, ``get_targets``, ``game_history_to_array``, ``store_search_stats``.
    """

    def __init__(self, action_space, game_history_length=200, config=None):
        """
        Overview:
            Init the ``GameHistory`` according to the provided arguments.
        Arguments:
             action_space (:obj:`int`): action space
            - game_history_length (:obj:`int`): the transition number of one ``GameHistory`` block
        """
        self.action_space = action_space
        self.game_history_length = game_history_length
        self.config = config

        self.frame_stack_num = config.frame_stack_num
        self.discount = config.discount
        self.action_space_size = config.action_space_size
        self.zero_obs_shape = (config.obs_shape[-2], config.obs_shape[-1], config.image_channel)

        self.obs_history = []
        self.action_history = []
        self.reward_history = []

        self.child_visit_history = []
        self.root_value_history = []

        self.action_mask_history = []
        self.to_play_history = []

        self.target_values = []
        self.target_rewards = []
        self.target_policies = []

    def __len__(self):
        return len(self.action_history)

    def init(self, init_observations):
        """
        Overview:
            Initialize the game history block using ``init_observations``,
            which is the previous ``frame_stack_num`` stacked frames.
        Arguments:
            - init_observations (:obj:`list`): list of the stack observations in the previous time steps.
        """
        self.obs_history = []
        self.action_history = []
        self.reward_history = []

        self.child_visit_history = []
        self.root_value_history = []

        self.action_mask_history = []
        self.to_play_history = []

        assert len(init_observations) == self.frame_stack_num

        for observation in init_observations:
            self.obs_history.append(copy.deepcopy(observation))

    def is_full(self):
        """
        Overview:
            check whether current game history block is full, i.e. larger than self.game_history_length
        """
        return self.__len__() >= self.game_history_length

    def legal_actions(self):
        return [_ for _ in range(self.action_space.n)]

    def append(self, action, obs, reward, action_mask=None, to_play=None):
        """
        Overview:
            append a transition tuple, including a_t, o_{t+1}, r_{t}, action_mask_{t}, to_play_{t}
        """
        self.action_history.append(action)
        self.obs_history.append(obs)
        self.reward_history.append(reward)

        self.action_mask_history.append(action_mask)
        self.to_play_history.append(to_play)

    def pad_over(self, next_block_observations, next_block_rewards, next_block_root_values, next_block_child_visits):
        """
        Overview:
            To make sure the correction of value targets, we need to add (o_t, r_t, etc) from the next history block
            , which is necessary for the bootstrapped values at the end states of this history block.
            Eg: len = 100; target value v_100 = r_100 + gamma^1 r_101 + ... + gamma^4 r_104 + gamma^5 v_105,
            but r_101, r_102, ... are from the next history block.
        Arguments:
            - next_block_observations (:obj:`list`):  list o_t from the next history block
            - next_block_rewards (:obj:`list`): list r_t from the next history block
            - next_block_root_values (:obj:`list`): list root values of MCTS from the next history block
            - next_block_child_visits (:obj:`list`): list root visit count distributions of MCTS from
            the next history block
        """
        assert len(next_block_observations) <= self.config.num_unroll_steps
        assert len(next_block_child_visits) <= self.config.num_unroll_steps
        assert len(next_block_root_values) <= self.config.num_unroll_steps + self.config.td_steps
        assert len(next_block_rewards) <= self.config.num_unroll_steps + self.config.td_steps - 1

        # NOTE: next block observation should start from (stacked_observation - 1) in next trajectory
        for observation in next_block_observations:
            self.obs_history.append(copy.deepcopy(observation))

        for reward in next_block_rewards:
            self.reward_history.append(reward)

        for value in next_block_root_values:
            self.root_value_history.append(value)

        for child_visits in next_block_child_visits:
            self.child_visit_history.append(child_visits)

    def obs(self, index, extra_len=0, padding=False):
        """
        Overview:
            To obtain an observation of correct format: o[t, t + stack frames + extra len]
        Arguments:
            - index: int time step
            - extra_len: int extra len of the obs frames
            - padding: bool True -> padding frames if (t + stack frames) are out of trajectory
        """
        frames = self.obs_history[index:index + self.frame_stack_num + extra_len]
        if padding:
            pad_len = self.frame_stack_num + extra_len - len(frames)
            if pad_len > 0:
                pad_frames = np.array([frames[-1] for _ in range(pad_len)])
                frames = np.concatenate((frames, pad_frames))
        if self.config.cvt_string:
            frames = [jpeg_data_decompressor(obs, self.config.gray_scale) for obs in frames]
        return frames

    def zero_obs(self):
        """
        Overview:
            return a zero obs frame
        """
        return [np.zeros(self.zero_obs_shape, dtype=np.uint8) for _ in range(self.frame_stack_num)]

    def step_obs(self):
        """
        Overview:
            return an observation in correct format for model inference
        """
        index = len(self.reward_history)
        frames = self.obs_history[index:index + self.frame_stack_num]
        if self.config.cvt_string:
            frames = [jpeg_data_decompressor(obs, self.config.gray_scale) for obs in frames]
        return frames

    def get_targets(self, i):
        """
        Overview:
            return the value/reward/policy targets at step i
        """
        return self.target_values[i], self.target_rewards[i], self.target_policies[i]

    def store_search_stats(self, visit_counts, root_value, idx: int = None):
        """
        Overview:
            store the visit count distributions and value of the root node after MCTS.
        """
        sum_visits = sum(visit_counts)
        if idx is None:
            self.child_visit_history.append([visit_count / sum_visits for visit_count in visit_counts])
            self.root_value_history.append(root_value)
        else:
            self.child_visit_history[idx] = [visit_count / sum_visits for visit_count in visit_counts]
            self.root_value_history[idx] = root_value

    def game_history_to_array(self):
        """
        Overview:
            post processing the data when a ``GameHistory`` block is full.
        Note:
        game_history element shape:
            e.g. game_history_length=20, stack=4, num_unroll_steps=5, td_steps=5

            obs:            game_history_length + stack + num_unroll_steps, 20+4+5
            action:         game_history_length -> 20
            reward:         game_history_length + num_unroll_steps + td_steps -1  20+5+5-1
            root_values:    game_history_length + num_unroll_steps + td_steps -> 20 +5+5
            child_visits：  game_history_length + num_unroll_steps -> 20+5
            to_play:        game_history_length -> 20
            action_mask:    game_history_length -> 20

        game_history_t:
            obs:  4       20        5
                 ----|----...----|-----|
        game_history_t+1:
            obs:               4       20        5
                             ----|----...----|-----|

        game_history_t:
            rew:     20        5      4
                 ----...----|------|-----|
        game_history_t+1:
            rew:             20        5    4
                        ----...----|-----|-----|
        """
        self.obs_history = np.array(self.obs_history)
        self.action_history = np.array(self.action_history)
        self.reward_history = np.array(self.reward_history)

        self.child_visit_history = np.array(self.child_visit_history)
        self.root_value_history = np.array(self.root_value_history)

        self.action_mask_history = np.array(self.action_mask_history)
        self.to_play_history = np.array(self.to_play_history)
