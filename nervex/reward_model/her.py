from typing import List, Dict, Any, Optional, Callable, Tuple
import copy
import numpy as np
import torch


class HerModel:
    """
    Overview:
        Hindsight Experience Replay model.

    .. note::
        - her_strategy (:obj:`str`): Type of strategy that HER uses, should be in ['final', 'future', 'episode']
        - her_replay_k (:obj:`int`): Number of new episodes generated by an original episode. (Not used in episodic HER)
        - episode_size (:obj:`int`): Sample how many episodes in one iteration.
        - sample_per_episode (:obj:`int`): How many new samples are generated from an episode.
    
    .. note::
        In HER, we require episode trajectory to change the goals. However, episode lengths are different and may have high variance. As a result, we **recommend** that you only use some transitions in the complete episode by specifying ``episode_size`` and ``sample_per_episode`` in config. Therefore, in one iteration, ``batch_size`` would be ``episode_size`` * ``sample_per_episode``.
    """

    def __init__(
            self,
            cfg: dict,
            cuda: bool = False,
    ) -> None:
        self._cuda = cuda and torch.cuda.is_available()
        self._device = 'cuda' if self._cuda else 'cpu'
        self._her_strategy = cfg.her_strategy
        assert self._her_strategy in ['final', 'future', 'episode']
        # `her_replay_k` may not be used in episodic HER, so default set to 1.
        self._her_replay_k = cfg.get('her_replay_k', 1)
        self._episode_size = cfg.get('episode_size', None)
        self._sample_per_episode = cfg.get('sample_per_episode', None)

    def estimate(
            self,
            episode: List[Dict[str, Any]],
            merge_func: Optional[Callable] = None,
            split_func: Optional[Callable] = None,
            goal_reward_func: Optional[Callable] = None
    ) -> List[Dict[str, Any]]:
        """
        Overview:
            Get HER processed episodes from original episodes.
        Arguments:
            - episode (:obj:`List[Dict[str, Any]]`): Episode list, each element is a transition.
            - merge_func (:obj:`Callable`): The merge function to use, default set to None. If None, \
                then use ``__her_default_merge_func``
            - split_func (:obj:`Callable`): The split function to use, default set to None. If None, \
                then use ``__her_default_split_func``
            - goal_reward_func (:obj:`Callable`): The goal_reward function to use, default set to None. If None, \
                then use ``__her_default_goal_reward_func``
        Returns:
            - new_episode (:obj:`List[Dict[str, Any]]`): the processed transitions
        """
        if merge_func is None:
            merge_func = HerModel.__her_default_merge_func
        if split_func is None:
            split_func = HerModel.__her_default_split_func
        if goal_reward_func is None:
            goal_reward_func = HerModel.__her_default_goal_reward_func
        new_episodes = [[] for _ in range(self._her_replay_k)]
        if self._sample_per_episode is None:
            # Use complete episode
            indices = range(len(episode))
        else:
            # Use some transitions in one episode
            indices = np.random.randint(0, len(episode), (self._sample_per_episode))
        for idx in indices:
            obs, _, _ = split_func(episode[idx]['obs'])
            next_obs, _, achieved_goal = split_func(episode[idx]['next_obs'])
            for k in range(self._her_replay_k):
                if self._her_strategy == 'final':
                    p_idx = -1
                elif self._her_strategy == 'episode':
                    p_idx = np.random.randint(0, len(episode))
                elif self._her_strategy == 'future':
                    p_idx = np.random.randint(idx, len(episode))
                _, _, new_desired_goal = split_func(episode[p_idx]['next_obs'])
                timestep = {
                    k: copy.deepcopy(v)
                    for k, v in episode[idx].items() if k not in ['obs', 'next_obs', 'reward']
                }
                timestep['obs'] = merge_func(obs, new_desired_goal)
                timestep['next_obs'] = merge_func(next_obs, new_desired_goal)
                timestep['reward'] = goal_reward_func(achieved_goal, new_desired_goal).to(self._device)
                new_episodes[k].append(timestep)
        return new_episodes

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
            Split the input into obs, desired goal, and achieved goal.
        Arguments:
            - x (:obj:`Any`): The input to split
        Returns:
            - obs (:obj:`torch.Tensor`): Original obs.
            - desired_goal (:obj:`torch.Tensor`): The final goal that wants to desired_goal
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

    @property
    def episode_size(self) -> int:
        return self._episode_size

    @property
    def sample_per_episode(self) -> int:
        return self._sample_per_episode
