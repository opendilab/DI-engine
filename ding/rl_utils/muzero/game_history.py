import numpy as np
from copy import deepcopy


class GameHistory:
    """
    Store only usefull information of a self-play game.
    """

    def __init__(self,config,max_length=200,):
        self.config = config

        self.to_play_history = []
        self.child_visits = []
        self.root_values = []
        self.max_length = max_length

        self.stacked_observations = config.stacked_observations
        self.reanalysed_predicted_root_values = None

        # For PER
        self.priorities = None
        self.game_priority = None

    def append(self, action, obs, reward):
        # append a transition tuple
        self.action_history.append(action)
        self.observation_history.append(obs)
        self.reward_history.append(reward)

    def store_search_stats(self, visit_counts, root_value, idx: int = None):
        # store the visit count distributions and value of the root node after MCTS
        sum_visits = sum(visit_counts)
        if idx is None:
            self.child_visits.append([visit_count / sum_visits for visit_count in visit_counts])
            self.root_values.append(root_value)
        else:
            self.child_visits[idx] = [visit_count / sum_visits for visit_count in visit_counts]
            self.root_values[idx] = root_value

    def get_stacked_observations(
        self, index, num_stacked_observations, action_space_size
    ):
        """
        Generate a new observation with the observation at the index position
        and num_stacked_observations past observations and actions stacked.
        """
        # Convert to positive index
        index = index % len(self.observation_history)

        stacked_observations = self.observation_history[index].copy()
        for past_observation_index in reversed(
            range(index - num_stacked_observations, index)
        ):
            if 0 <= past_observation_index:
                previous_observation = np.concatenate(
                    (
                        self.observation_history[past_observation_index],
                        [
                            np.ones_like(stacked_observations[0])
                            * self.action_history[past_observation_index + 1]
                            / action_space_size
                        ],
                    )
                )
            else:
                previous_observation = np.concatenate(
                    (
                        np.zeros_like(self.observation_history[index]),
                        [np.zeros_like(stacked_observations[0])],
                    )
                )

            stacked_observations = np.concatenate(
                (stacked_observations, previous_observation)
            )

        return stacked_observations

    def __len__(self):
        return len(self.actions)

    def is_full(self):
        # history block is full
        return self.__len__() >= self.max_length