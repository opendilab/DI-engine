from typing import TYPE_CHECKING, Callable, List, Tuple, Any
from easydict import EasyDict
from functools import reduce
import torch
import treetensor.torch as ttorch
from ding.envs import BaseEnvManager
from ding.policy import Policy
from ding.framework import task
from collections import namedtuple
from ding.utils import EasyTimer, dicts_to_lists
from ding.torch_utils import to_tensor, to_ndarray
from ding.worker.collector.base_serial_collector import to_tensor_transitions

if TYPE_CHECKING:
    from ding.framework import OnlineRLContext


class TransitionList:

    def __init__(self, env_num: int) -> None:
        self.env_num = env_num
        self._transitions = [[] for _ in range(env_num)]
        self._done_idx = [[] for _ in range(env_num)]

    def append(self, env_id: int, transition: Any) -> None:
        self._transitions[env_id].append(transition)
        if transition.done:
            self._done_idx[env_id].append(len(self._transitions[env_id]))

    def to_trajectories(self) -> Tuple[List[Any], List[int]]:
        trajectories = sum(self._transitions, [])
        lengths = [len(t) for t in self._transitions]
        trajectory_end_idx = [reduce(lambda x, y: x + y, lengths[:i + 1]) for i in range(len(lengths))]
        trajectory_end_idx = [t - 1 for t in trajectory_end_idx]
        return trajectories, trajectory_end_idx

    def to_episodes(self) -> List[List[Any]]:
        episodes = []
        for env_id in range(self.env_num):
            last_idx = 0
            for done_idx in self._done_idx[env_id]:
                episodes.append(self._transitions[env_id][last_idx:done_idx])
                last_idx = done_idx
        return episodes

    def clear(self):
        for item in self._transitions:
            item.clear()
        for item in self._done_idx:
            item.clear()


def inferencer(cfg: EasyDict, policy: Policy, env: BaseEnvManager) -> Callable:
    """
    Overview:
        The middleware that executes the inference process.
    Arguments:
        - cfg (:obj:`EasyDict`): Config.
        - policy (:obj:`Policy`): The policy to be inferred.
        - env (:obj:`BaseEnvManager`): The env where the inference process is performed. \
            The env.ready_obs (:obj:`tnp.array`) will be used as model input.
    """

    env.seed(cfg.seed)

    def _inference(ctx: "OnlineRLContext"):
        """
        Output of ctx:
            - obs (:obj:`Dict[Tensor]`): The input states fed into the model.
            - action: (:obj:`List[np.ndarray]`): The inferred actions listed by env_id.
            - inference_output (:obj:`Dict[int, Dict]`): The dict that contains env_id (int) \
                and inference result (Dict).
        """

        if env.closed:
            env.launch()

        obs = ttorch.as_tensor(env.ready_obs).to(dtype=ttorch.float32)
        ctx.obs = obs
        # TODO mask necessary rollout

        obs = {i: obs[i] for i in range(obs.shape[0])}  # TBD
        inference_output = policy.forward(obs, **ctx.collect_kwargs)
        ctx.action = [v['action'].numpy() for v in inference_output.values()]  # TBD
        ctx.inference_output = inference_output

    return _inference

class BattleCollector():
    def __init__(self, cfg: EasyDict, policy: List[namedtuple] = None, env: BaseEnvManager = None):
        self._cfg = cfg
        self._transform_obs = cfg.transform_obs
        self._timer = EasyTimer()

        # TODO(zms) call self.reset() to reset policy and env
        self._policy = policy
        self._env = env 


    def __call__(self, ctx: "OnlineRLContext") -> None:
        """
        Input of ctx:
            - n_episode (:obj:`int`): the number of collecting data episode
            - train_iter (:obj:`int`): the number of training iteration
            - policy_kwargs (:obj:`dict`): the keyword args for policy forward
        Output of ctx:
            -  return_data (:obj:`Tuple[List, List]`): A tuple with training sample(data) and episode info, \
                the former is a list containing collected episodes if not get_train_sample, \
                otherwise, return train_samples split by unroll_len.
        """
        
        if ctx.n_episode is None:
            ### TODO(zms): self._default_n_episode comes from self.reset_policy()
            if self._default_n_episode is None:
                raise RuntimeError("Please specify collect n_episode")
            else:
                ctx.n_episode = self._default_n_episode
        ### TODO(zms): self._env_num comes from self.reset_env()
        assert ctx.n_episode >= self._env_num, "Please make sure n_episode >= env_num"

        if ctx.policy_kwargs is None:
            ctx.policy_kwargs = {}
        
        collected_episode = 0
        return_data = [[] for _ in range(2)]
        return_info = [[] for _ in range(2)]
        ready_env_id = set()
        remain_episode = ctx.n_episode

        while True:
            with self._timer:
                # Get current env obs.
                obs = self._env.ready_obs
                new_available_env_id = set(obs.keys()).difference(ready_env_id)
                ready_env_id = ready_env_id.union(set(list(new_available_env_id)[:remain_episode]))
                remain_episode -= min(len(new_available_env_id), remain_episode)
                obs = {env_id: obs[env_id] for env_id in ready_env_id}

                # Policy forward.
                ### TODO(zms): self._obs_pool comes from self.reset
                self._obs_pool.update(obs)
                if self._transform_obs:
                    obs = to_tensor(obs, dtype=torch.float32)
                obs = dicts_to_lists(obs)
                policy_output = [p.forward(obs[i], **ctx.policy_kwargs) for i, p in enumerate(self._policy)]

                ### TODO(zms): self._policy_output_pool comes from self.reset
                self._policy_output_pool.update(policy_output)
                # Interact with env.
                actions = {}
                for env_id in ready_env_id:
                    actions[env_id] = []
                    for output in policy_output:
                        actions[env_id].append(output[env_id]['action'])
                actions = to_ndarray(actions)
                timesteps = self._env.step(actions)

            # TODO(nyz) this duration may be inaccurate in async env
            interaction_duration = self._timer.value / len(timesteps)

            # TODO(nyz) vectorize this for loop
            for env_id, timestep in timesteps.items():
                # TODO(zms): self._env_info comes from self.reset()
                self._env_info[env_id]['step'] += 1
                # TODO(zms): self._total_envstep_count comes from self.reset()
                self._total_envstep_count += 1
                with self._timer:
                    for policy_id, policy in enumerate(self._policy):
                        policy_timestep_data = [d[policy_id] if not isinstance(d, bool) else d for d in timestep]
                        policy_timestep = type(timestep)(*policy_timestep_data)
                        transition = self._policy[policy_id].process_transition(
                            # TODO(zms): self._policy_output_pool comes from self.reset()
                            self._obs_pool[env_id][policy_id], self._policy_output_pool[env_id][policy_id],
                            policy_timestep
                        )
                        transition['collect_iter'] = ctx.train_iter
                        # TODO(zms): self._traj_buffer comes from self.reset()
                        self._traj_buffer[env_id][policy_id].append(transition)
                        # prepare data
                        if timestep.done:
                            # TODO(zms): to get rid of collector, "to_tensor_transitions" function should be removed and must be somewhere else.
                            transitions = to_tensor_transitions(self._traj_buffer[env_id][policy_id])
                            if self._cfg.get_train_sample:
                                train_sample = self._policy[policy_id].get_train_sample(transitions)
                                return_data[policy_id].extend(train_sample)
                            else:
                                return_data[policy_id].append(transitions)
                            self._traj_buffer[env_id][policy_id].clear()

                self._env_info[env_id]['time'] += self._timer.value + interaction_duration

                # If env is done, record episode info and reset
                if timestep.done:
                    # TODO(zms): self._total_episode_count comes from self.reset()
                    self._total_episode_count += 1
                    info = {
                        'reward0': timestep.info[0]['final_eval_reward'],
                        'reward1': timestep.info[1]['final_eval_reward'],
                        'time': self._env_info[env_id]['time'],
                        'step': self._env_info[env_id]['step'],
                    }
                    collected_episode += 1
                    # TODO(zms): self._episode_info comes from self.reset()
                    self._episode_info.append(info)
                    for i, p in enumerate(self._policy):
                        p.reset([env_id])
                    # TODO(zms): define self._reset_stat 
                    self._reset_stat(env_id)
                    ready_env_id.remove(env_id)
                    for policy_id in range(2):
                        return_info[policy_id].append(timestep.info[policy_id])
            if collected_episode >= ctx.n_episode:
                break
        # log
        ### TODO: how to deal with log here?
        # self._output_log(ctx.train_iter)
        ctx.return_data = return_data
        ctx.return_info = return_info



def rolloutor(cfg: EasyDict, policy: Policy, env: BaseEnvManager, transitions: TransitionList) -> Callable:
    """
    Overview:
        The middleware that executes the transition process in the env.
    Arguments:
        - cfg (:obj:`EasyDict`): Config.
        - policy (:obj:`Policy`): The policy to be used during transition.
        - env (:obj:`BaseEnvManager`): The env for the collection, the BaseEnvManager object or \
                its derivatives are supported.
        - transitions (:obj:`TransitionList`): The transition information which will be filled \
            in this process, including `obs`, `next_obs`, `action`, `logit`, `value`, `reward` \
            and `done`.
    """

    env_episode_id = [_ for _ in range(env.env_num)]
    current_id = env.env_num

    def _rollout(ctx: "OnlineRLContext"):
        """
        Input of ctx:
            - action: (:obj:`List[np.ndarray]`): The inferred actions from previous inference process.
            - obs (:obj:`Dict[Tensor]`): The states fed into the transition dict.
            - inference_output (:obj:`Dict[int, Dict]`): The inference results to be fed into the \
                transition dict.
            - train_iter (:obj:`int`): The train iteration count to be fed into the transition dict.
            - env_step (:obj:`int`): The count of env step, which will increase by 1 for a single \
                transition call.
            - env_episode (:obj:`int`): The count of env episode, which will increase by 1 if the \
                trajectory stops.
        """

        nonlocal current_id
        timesteps = env.step(ctx.action)
        ctx.env_step += len(timesteps)
        timesteps = [t.tensor() for t in timesteps]
        # TODO abnormal env step
        for i, timestep in enumerate(timesteps):
            transition = policy.process_transition(ctx.obs[i], ctx.inference_output[i], timestep)
            transition = ttorch.as_tensor(transition)  # TBD
            transition.collect_train_iter = ttorch.as_tensor([ctx.train_iter])
            transition.env_data_id = ttorch.as_tensor([env_episode_id[timestep.env_id]])
            transitions.append(timestep.env_id, transition)
            if timestep.done:
                policy.reset([timestep.env_id])
                env_episode_id[timestep.env_id] = current_id
                current_id += 1
                ctx.env_episode += 1
        # TODO log

    return _rollout
