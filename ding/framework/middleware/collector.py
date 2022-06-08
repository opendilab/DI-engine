from distutils.log import info
from easydict import EasyDict
from ding.policy import Policy, get_random_policy
from ding.envs import BaseEnvManager
from ding.framework import task
from .functional import inferencer, rolloutor, TransitionList, battle_inferencer, battle_rolloutor
from typing import Dict

# if TYPE_CHECKING:
from ding.framework import OnlineRLContext, BattleContext


class BattleEpisodeCollector:

    def __init__(self, cfg: EasyDict, env: BaseEnvManager, n_rollout_samples: int, model_dict: Dict, all_policies: Dict, agent_num: int):
        self.cfg = cfg
        self.end_flag = False
        # self._reset(env)
        self.env = env
        self.env_num = self.env.env_num

        self.total_envstep_count = 0
        self.n_rollout_samples = n_rollout_samples
        self.model_dict = model_dict
        self.all_policies = all_policies
        self.agent_num = agent_num

        self._battle_inferencer = task.wrap(battle_inferencer(self.cfg, self.env))
        self._transitions_list = [TransitionList(self.env.env_num) for _ in range(self.agent_num)]
        self._battle_rolloutor = task.wrap(battle_rolloutor(self.cfg, self.env, self._transitions_list))

    def __del__(self) -> None:
        """
        Overview:
            Execute the close command and close the collector. __del__ is automatically called to \
                destroy the collector instance when the collector finishes its work
        """
        if self.end_flag:
            return
        self.end_flag = True
        self.env.close()
    
    def _update_policies(self, job) -> None:
        job_player_id_list = [player.player_id for player in job.players] 

        for player_id in job_player_id_list:
            if self.model_dict.get(player_id) is None:
                continue
            else:
                learner_model = self.model_dict.get(player_id)
                policy = self.all_policies.get(player_id)
                assert policy, "for player{}, policy should have been initialized already"
                # update policy model
                policy.load_state_dict(learner_model.state_dict)
                self.model_dict[player_id] = None

    def __call__(self, ctx: "BattleContext") -> None:
        """
        Input of ctx:
            - n_episode (:obj:`int`): the number of collecting data episode
            - train_iter (:obj:`int`): the number of training iteration
            - collect_kwargs (:obj:`dict`): the keyword args for policy forward
        Output of ctx:
            -  ctx.train_data (:obj:`Tuple[List, List]`): A tuple with training sample(data) and episode info, \
                the former is a list containing collected episodes if not get_train_sample, \
                otherwise, return train_samples split by unroll_len.
        """
        ctx.total_envstep_count = self.total_envstep_count
        old = ctx.env_episode
        while True:
            self._update_policies(ctx.job)
            self._battle_inferencer(ctx)
            self._battle_rolloutor(ctx)

            self.total_envstep_count = ctx.total_envstep_count

            if (self.n_rollout_samples > 0 and ctx.env_episode - old >= self.n_rollout_samples) or ctx.env_episode >= ctx.n_episode:
                for transitions in self._transitions_list:
                    ctx.episodes.append(transitions.to_episodes())
                    transitions.clear()
                if ctx.env_episode >= ctx.n_episode:
                    ctx.job_finish = True
                break


class BattleStepCollector:

    def __init__(self, cfg: EasyDict, env: BaseEnvManager, n_rollout_samples: int, model_dict: Dict, all_policies: Dict, agent_num: int):
        self.cfg = cfg
        self.end_flag = False
        # self._reset(env)
        self.env = env
        self.env_num = self.env.env_num

        self.total_envstep_count = 0
        self.n_rollout_samples = n_rollout_samples
        self.model_dict = model_dict
        self.all_policies = all_policies
        self.agent_num = agent_num

        self._battle_inferencer = task.wrap(battle_inferencer(self.cfg, self.env))
        self._transitions_list = [TransitionList(self.env.env_num) for _ in range(self.agent_num)]
        self._battle_rolloutor = task.wrap(battle_rolloutor(self.cfg, self.env, self._transitions_list))

    def __del__(self) -> None:
        """
        Overview:
            Execute the close command and close the collector. __del__ is automatically called to \
                destroy the collector instance when the collector finishes its work
        """
        if self.end_flag:
            return
        self.end_flag = True
        self.env.close()
    
    def _update_policies(self, job) -> None:
        job_player_id_list = [player.player_id for player in job.players] 

        for player_id in job_player_id_list:
            if self.model_dict.get(player_id) is None:
                continue
            else:
                learner_model = self.model_dict.get(player_id)
                policy = self.all_policies.get(player_id)
                assert policy, "for player{}, policy should have been initialized already"
                # update policy model
                policy.load_state_dict(learner_model.state_dict)
                self.model_dict[player_id] = None

    def __call__(self, ctx: "BattleContext") -> None:
        """
        Input of ctx:
            - n_episode (:obj:`int`): the number of collecting data episode
            - train_iter (:obj:`int`): the number of training iteration
            - collect_kwargs (:obj:`dict`): the keyword args for policy forward
        Output of ctx:
            -  ctx.train_data (:obj:`Tuple[List, List]`): A tuple with training sample(data) and episode info, \
                the former is a list containing collected episodes if not get_train_sample, \
                otherwise, return train_samples split by unroll_len.
        """
        ctx.total_envstep_count = self.total_envstep_count
        old = ctx.env_step
        
        while True:
            self._update_policies(ctx.job)
            self._battle_inferencer(ctx)
            self._battle_rolloutor(ctx)

            self.total_envstep_count = ctx.total_envstep_count

            if (self.n_rollout_samples > 0 and ctx.env_step - old >= self.n_rollout_samples) or ctx.env_step >= ctx.n_sample * ctx.unroll_len:
                for transitions in self._transitions_list:
                    trajectories, trajectory_end_idx = transitions.to_trajectories()
                    ctx.trajectories_list.append(trajectories)
                    ctx.trajectory_end_idx_list.append(trajectory_end_idx)
                    transitions.clear()
                if ctx.env_step >= ctx.n_sample * ctx.unroll_len:
                    ctx.job_finish = True
                break


class StepCollector:
    """
    Overview:
        The class of the collector running by steps, including model inference and transition \
            process. Use the `__call__` method to execute the whole collection process.
    """

    def __init__(self, cfg: EasyDict, policy, env: BaseEnvManager, random_collect_size: int = 0) -> None:
        """
        Arguments:
            - cfg (:obj:`EasyDict`): Config.
            - policy (:obj:`Policy`): The policy to be collected.
            - env (:obj:`BaseEnvManager`): The env for the collection, the BaseEnvManager object or \
                its derivatives are supported.
            - random_collect_size (:obj:`int`): The count of samples that will be collected randomly, \
                typically used in initial runs.
        """
        self.cfg = cfg
        self.env = env
        self.policy = policy
        self.random_collect_size = random_collect_size
        self._transitions = TransitionList(self.env.env_num)
        self._inferencer = task.wrap(inferencer(cfg, policy, env))
        self._rolloutor = task.wrap(rolloutor(cfg, policy, env, self._transitions))

    def __call__(self, ctx: "OnlineRLContext") -> None:
        """
        Overview:
            An encapsulation of inference and rollout middleware. Stop when completing \
                the target number of steps.
        Input of ctx:
            - env_step (:obj:`int`): The env steps which will increase during collection.
        """
        old = ctx.env_step
        if self.random_collect_size > 0 and old < self.random_collect_size:
            target_size = self.random_collect_size - old
            random_policy = get_random_policy(self.cfg, self.policy, self.env)
            current_inferencer = task.wrap(inferencer(self.cfg, random_policy, self.env))
        else:
            # compatible with old config, a train sample = unroll_len step
            target_size = self.cfg.policy.collect.n_sample * self.cfg.policy.collect.unroll_len
            current_inferencer = self._inferencer

        while True:
            current_inferencer(ctx)
            self._rolloutor(ctx)
            if ctx.env_step - old >= target_size:
                ctx.trajectories, ctx.trajectory_end_idx = self._transitions.to_trajectories()
                self._transitions.clear()
                break


class EpisodeCollector:
    """
    Overview:
        The class of the collector running by episodes, including model inference and transition \
            process. Use the `__call__` method to execute the whole collection process.
    """

    def __init__(self, cfg: EasyDict, policy, env: BaseEnvManager, random_collect_size: int = 0) -> None:
        """
        Arguments:
            - cfg (:obj:`EasyDict`): Config.
            - policy (:obj:`Policy`): The policy to be collected.
            - env (:obj:`BaseEnvManager`): The env for the collection, the BaseEnvManager object or \
                its derivatives are supported.
            - random_collect_size (:obj:`int`): The count of samples that will be collected randomly, \
                typically used in initial runs.
        """
        self.cfg = cfg
        self.env = env
        self.policy = policy
        self.random_collect_size = random_collect_size
        self._transitions = TransitionList(self.env.env_num)
        self._inferencer = task.wrap(inferencer(cfg, policy, env))
        self._rolloutor = task.wrap(rolloutor(cfg, policy, env, self._transitions))

    def __call__(self, ctx: "OnlineRLContext") -> None:
        """
        Overview:
            An encapsulation of inference and rollout middleware. Stop when completing the \
                target number of episodes.
        Input of ctx:
            - env_episode (:obj:`int`): The env env_episode which will increase during collection.
        """
        old = ctx.env_episode
        if self.random_collect_size > 0 and old < self.random_collect_size:
            target_size = self.random_collect_size - old
            random_policy = get_random_policy(self.cfg, self.policy, self.env)
            current_inferencer = task.wrap(inferencer(self.cfg, random_policy, self.env))
        else:
            target_size = self.cfg.policy.collect.n_episode
            current_inferencer = self._inferencer

        while True:
            current_inferencer(ctx)
            self._rolloutor(ctx)
            if ctx.env_episode - old >= target_size:
                ctx.episodes = self._transitions.to_episodes()
                self._transitions.clear()
                break


# TODO battle collector
