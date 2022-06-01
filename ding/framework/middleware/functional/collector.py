from typing import TYPE_CHECKING, Optional, Callable, List, Tuple, Any
from easydict import EasyDict
from functools import reduce
import treetensor.torch as ttorch
from ding.envs import BaseEnvManager
from ding.policy import Policy
import torch
from ding.utils import dicts_to_lists
from ding.torch_utils import to_tensor, to_ndarray
from ding.worker.collector.base_serial_collector import CachePool, TrajBuffer, to_tensor_transitions

# if TYPE_CHECKING:
from ding.framework import OnlineRLContext, BattleContext


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


def policy_resetter(env_num: int):

    def _policy_resetter(ctx: "BattleContext"):
        if ctx.current_policies is not None:
            assert len(ctx.current_policies) > 1, "battle collector needs more than 1 policies"
            ctx._default_n_episode = ctx.current_policies[0].get_attribute('cfg').collect.get('n_episode', None)
            ctx.agent_num = len(ctx.current_policies)
            ctx.traj_len = float("inf")
            # traj_buffer is {env_id: {policy_id: TrajBuffer}}, is used to store traj_len pieces of transitions
            ctx.traj_buffer = {
                env_id: {policy_id: TrajBuffer(maxlen=ctx.traj_len)
                         for policy_id in range(ctx.agent_num)}
                for env_id in range(env_num)
            }

            for p in ctx.current_policies:
                p.reset()
        else:
            raise RuntimeError('ctx.current_policies should not be None')

    return _policy_resetter


def battle_inferencer(cfg: EasyDict, env: BaseEnvManager, obs_pool: CachePool, policy_output_pool: CachePool):

    def _battle_inferencer(ctx: "BattleContext"):
        # Get current env obs.
        obs = env.ready_obs
        new_available_env_id = set(obs.keys()).difference(ctx.ready_env_id)
        ctx.ready_env_id = ctx.ready_env_id.union(set(list(new_available_env_id)[:ctx.remain_episode]))
        ctx.remain_episode -= min(len(new_available_env_id), ctx.remain_episode)
        obs = {env_id: obs[env_id] for env_id in ctx.ready_env_id}

        # Policy forward.
        obs_pool.update(obs)
        if cfg.transform_obs:
            obs = to_tensor(obs, dtype=torch.float32)
        obs = dicts_to_lists(obs)
        policy_output = [p.forward(obs[i], **ctx.policy_kwargs) for i, p in enumerate(ctx.current_policies)]
        policy_output_pool.update(policy_output)

        # Interact with env.
        actions = {}
        for env_id in ctx.ready_env_id:
            actions[env_id] = []
            for output in policy_output:
                actions[env_id].append(output[env_id]['action'])
        ctx.actions = to_ndarray(actions)

    return _battle_inferencer


def battle_rolloutor(cfg: EasyDict, env: BaseEnvManager, obs_pool: CachePool, policy_output_pool: CachePool):

    def _battle_rolloutor(ctx: "BattleContext"):
        timesteps = env.step(ctx.actions)
        for env_id, timestep in timesteps.items():
            ctx.envstep += 1
            for policy_id, _ in enumerate(ctx.current_policies):
                policy_timestep_data = [d[policy_id] if not isinstance(d, bool) else d for d in timestep]
                policy_timestep = type(timestep)(*policy_timestep_data)
                transition = ctx.current_policies[policy_id].process_transition(
                    obs_pool[env_id][policy_id], policy_output_pool[env_id][policy_id], policy_timestep
                )
                transition['collect_iter'] = ctx.train_iter
                ctx.traj_buffer[env_id][policy_id].append(transition)
                # If env is done, prepare data
                if timestep.done:
                    transitions = to_tensor_transitions(ctx.traj_buffer[env_id][policy_id])
                    if cfg.get_train_sample:
                        train_sample = ctx.current_policies[policy_id].get_train_sample(transitions)
                        ctx.train_data[policy_id].extend(train_sample)
                    else:
                        ctx.train_data[policy_id].append(transitions)
                    ctx.traj_buffer[env_id][policy_id].clear()

            if timestep.done:
                ctx.collected_episode += 1
                for i, p in enumerate(ctx.current_policies):
                    p.reset([env_id])
                for i in range(ctx.agent_num):
                    ctx.traj_buffer[env_id][i].clear()
                obs_pool.reset(env_id)
                policy_output_pool.reset(env_id)
                ctx.ready_env_id.remove(env_id)
                for policy_id in range(ctx.agent_num):
                    ctx.episode_info[policy_id].append(timestep.info[policy_id])

    return _battle_rolloutor
