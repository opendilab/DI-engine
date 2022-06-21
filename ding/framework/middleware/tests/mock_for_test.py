from typing import TYPE_CHECKING, Union, Any, List, Callable, Dict, Optional
from collections import namedtuple
import random
import torch
import treetensor.numpy as tnp
from easydict import EasyDict
from unittest.mock import Mock

from ding.torch_utils import to_device
from ding.league.player import PlayerMeta
from ding.league.v2 import BaseLeague, Job
from ding.framework.storage import FileStorage
from dizoo.distar.envs.distar_env import DIStarEnv
from dizoo.distar.policy.distar_policy import DIStarPolicy
from ding.envs import BaseEnvManager
import treetensor.torch as ttorch
from ding.envs import BaseEnvTimestep
from distar.agent.default.lib.features import Features

if TYPE_CHECKING:
    from ding.framework import BattleContext

obs_dim = [2, 2]
action_space = 1
env_num = 2

CONFIG = dict(
    seed=0,
    policy=dict(
        learn=dict(
            update_per_collect=4,
            batch_size=8,
            learner=dict(hook=dict(log_show_after_iter=10), ),
        ),
        collect=dict(
            n_sample=16,
            unroll_len=1,
            n_episode=16,
        ),
        eval=dict(evaluator=dict(eval_freq=10), ),
        other=dict(eps=dict(
            type='exp',
            start=0.95,
            end=0.1,
            decay=10000,
        ), ),
    ),
    env=dict(
        n_evaluator_episode=5,
        stop_value=2.0,
    ),
)
CONFIG = EasyDict(CONFIG)


class MockPolicy(Mock):

    def __init__(self) -> None:
        super(MockPolicy, self).__init__()
        self.action_space = action_space
        self.obs_dim = obs_dim

    def reset(self, data_id: Optional[List[int]] = None) -> None:
        return

    def forward(self, data: dict, **kwargs) -> dict:
        res = {}
        for i, v in data.items():
            res[i] = {'action': torch.sum(v)}
        return res

    def process_transition(self, obs: Any, model_output: dict, timestep: namedtuple) -> dict:
        transition = {
            'obs': torch.rand(self.obs_dim),
            'next_obs': torch.rand(self.obs_dim),
            'action': torch.zeros(self.action_space),
            'logit': 1.0,
            'value': 2.0,
            'reward': 0.1,
            'done': True,
        }
        return transition


class MockEnv(Mock):

    def __init__(self) -> None:
        super(MockEnv, self).__init__()
        self.env_num = env_num
        self.obs_dim = obs_dim
        self.closed = False
        self._reward_grow_indicator = 1

    @property
    def ready_obs(self) -> tnp.array:
        return tnp.stack([
            torch.zeros(self.obs_dim),
            torch.ones(self.obs_dim),
        ])

    def seed(self, seed: Union[Dict[int, int], List[int], int], dynamic_seed: bool = None) -> None:
        return

    def launch(self, reset_param: Optional[Dict] = None) -> None:
        return

    def reset(self, reset_param: Optional[Dict] = None) -> None:
        return

    def step(self, actions: tnp.ndarray) -> List[tnp.ndarray]:
        timesteps = []
        for i in range(self.env_num):
            timestep = dict(
                obs=torch.rand(self.obs_dim),
                reward=1.0,
                done=True,
                info={'final_eval_reward': self._reward_grow_indicator * 1.0},
                env_id=i,
            )
            timesteps.append(tnp.array(timestep))
        self._reward_grow_indicator += 1  # final_eval_reward will increase as step method is called
        return timesteps


class MockHerRewardModel(Mock):

    def __init__(self) -> None:
        super(MockHerRewardModel, self).__init__()
        self.episode_size = 8
        self.episode_element_size = 4

    def estimate(self, episode: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [[episode[0] for _ in range(self.episode_element_size)]]


class MockLeague(BaseLeague):

    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self.update_payoff_cnt = 0
        self.update_active_player_cnt = 0
        self.create_historical_player_cnt = 0
        self.get_job_info_cnt = 0

    def update_payoff(self, job):
        self.update_payoff_cnt += 1

    def update_active_player(self, meta):
        self.update_active_player_cnt += 1

    def create_historical_player(self, meta):
        self.create_historical_player_cnt += 1

    def get_job_info(self, player_id):
        self.get_job_info_cnt += 1
        other_players = [i for i in self.active_players_ids if i != player_id]
        another_palyer = random.choice(other_players)
        return Job(
            launch_player=player_id,
            players=[
                PlayerMeta(player_id=player_id, checkpoint=FileStorage(path=None), total_agent_step=0),
                PlayerMeta(player_id=another_palyer, checkpoint=FileStorage(path=None), total_agent_step=0)
            ]
        )


class MockLogger():

    def add_scalar(*args):
        pass

    def close(*args):
        pass

    def flush(*args):
        pass


class DIStarMockPolicy(DIStarPolicy):

    def _forward_learn(self, data: Dict[str, Any]) -> Dict[str, Any]:
        print("Call forward_learn:")
        return super()._forward_learn(data)

    def _forward_collect(self, data: Dict[int, Any]) -> Dict[int, Any]:
        return DIStarEnv.random_action(data)



class DIstarCollectMode:

    def __init__(self) -> None:
        self._cfg = EasyDict(dict(collect=dict(n_episode=1)))
        self._feature = None
        self._race = 'zerg'

    def load_state_dict(self, state_dict):
        return

    def get_attribute(self, name: str) -> Any:
        if hasattr(self, '_get_' + name):
            return getattr(self, '_get_' + name)()
        elif hasattr(self, '_' + name):
            return getattr(self, '_' + name)
        else:
            raise NotImplementedError

    def reset(self, data_id: Optional[List[int]] = None) -> None:
        pass

    def _pre_process(self, obs):
        agent_obs = self._feature.transform_obs(obs['raw_obs'], padding_spatial=True)
        self._game_info = agent_obs.pop('game_info')
        self._game_step = self._game_info['game_loop']

        last_selected_units = torch.zeros(agent_obs['entity_num'], dtype=torch.int8)
        last_targeted_unit = torch.zeros(agent_obs['entity_num'], dtype=torch.int8)

        agent_obs['entity_info']['last_selected_units'] = last_selected_units
        agent_obs['entity_info']['last_targeted_unit'] = last_targeted_unit

        self._observation = agent_obs

    def forward(self, policy_obs: Dict[int, Any]) -> Dict[int, Any]:
        # print("Call forward_collect:")
        self._pre_process(policy_obs)
        return_data = {}
        return_data['action'] = DIStarEnv.random_action(policy_obs)
        return_data['logit'] = [1]
        return_data['value'] = [0]

        return return_data

    def process_transition(self, obs: Any, model_output: dict, timestep: namedtuple) -> dict:
        next_obs = timestep.obs
        reward = timestep.reward
        done = timestep.done
        agent_obs = self._observation
        step_data = {
            'obs': {
                'map_name': 'KingsCove',
                'spatial_info': agent_obs['spatial_info'],
                # 'spatial_info_ref': spatial_info_ref,
                'entity_info': agent_obs['entity_info'],
                'scalar_info': agent_obs['scalar_info'],
                'entity_num': agent_obs['entity_num'],
                'step': torch.tensor(self._game_step, dtype=torch.float)
            },
            'next_obs': {},
            'logit': model_output['logit'],
            'action': model_output['action'],
            'value': model_output['value'],
            # 'successive_logit': deepcopy(teacher_output['logit']),
            'reward': reward,
            'done': done
        }
        return step_data


class DIStarMockPolicyCollect:

    def __init__(self):

        self.collect_mode = DIstarCollectMode()


def battle_inferencer_for_distar(cfg: EasyDict, env: BaseEnvManager):

    def _battle_inferencer(ctx: "BattleContext"):

        if env.closed:
            env.launch()

        # TODO: Just for distar
        races = ['zerg', 'zerg']
        for policy_index, p in enumerate(ctx.current_policies):
            if p._feature is None:
                p._feature = Features(env._envs[0].game_info[policy_index], env.ready_obs[0][policy_index]['raw_obs'])
                p._race = races[policy_index]

        # Get current env obs.
        obs = env.ready_obs
        # the role of remain_episode is to mask necessary rollouts, avoid processing unnecessary data
        new_available_env_id = set(obs.keys()).difference(ctx.ready_env_id)
        ctx.ready_env_id = ctx.ready_env_id.union(set(list(new_available_env_id)[:ctx.remain_episode]))
        ctx.remain_episode -= min(len(new_available_env_id), ctx.remain_episode)
        obs = {env_id: obs[env_id] for env_id in ctx.ready_env_id}

        ctx.obs = obs

        # Policy forward.
        inference_output = {}
        actions = {}
        for env_id in ctx.ready_env_id:
            observations = obs[env_id]
            inference_output[env_id] = {}
            actions[env_id] = {}
            for policy_id, policy_obs in observations.items():
                output = ctx.current_policies[policy_id].forward(policy_obs)
                inference_output[env_id][policy_id] = output
                actions[env_id][policy_id] = output['action']
        ctx.inference_output = inference_output
        ctx.actions = actions

    return _battle_inferencer


def battle_rolloutor_for_distar(cfg: EasyDict, env: BaseEnvManager, transitions_list: List):

    def _battle_rolloutor(ctx: "BattleContext"):
        timesteps = env.step(ctx.actions)
        # TODO: change this part to only modify the part of current episode, not influence previous episode
        error_env_id_list = []
        for env_id, timestep in timesteps.items():
            if timestep.info.get('step_error'):
                error_env_id_list.append(env_id)
                ctx.total_envstep_count -= transitions_list[0].length(env_id)
                ctx.env_step -= transitions_list[0].length(env_id)
                for transitions in transitions_list:
                    transitions.clear_env_transitions(env_id)
        for error_env_id in error_env_id_list:
            del timesteps[error_env_id]

        ctx.total_envstep_count += len(timesteps)
        ctx.env_step += len(timesteps)
        for env_id, timestep in timesteps.items():
            for policy_id in ctx.obs[env_id].keys():
                policy_timestep = BaseEnvTimestep(
                    obs=timestep.obs.get(policy_id) if timestep.obs.get(policy_id) is not None else None,
                    reward=timestep.reward[policy_id],
                    done=timestep.done,
                    info={}
                )
                transition = ctx.current_policies[policy_id].process_transition(
                    ctx.obs[env_id][policy_id], ctx.inference_output[env_id][policy_id], policy_timestep
                )
                transition = EasyDict(transition)
                transition.collect_train_iter = ttorch.as_tensor([ctx.train_iter])
                transitions_list[policy_id].append(env_id, transition)
                if timestep.done:
                    ctx.current_policies[policy_id].reset([env_id])
                    ctx.episode_info[policy_id].append(timestep.info[policy_id])

            if timestep.done:
                ctx.ready_env_id.remove(env_id)
                ctx.env_episode += 1

    return _battle_rolloutor
