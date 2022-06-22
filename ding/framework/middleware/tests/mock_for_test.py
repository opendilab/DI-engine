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
from dizoo.distar.envs.fake_data import rl_step_data

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

    def forward(self, policy_obs: Dict[int, Any]) -> Dict[int, Any]:
        # print("Call forward_collect:")
        return_data = {}
        return_data['action'] = DIStarEnv.random_action(policy_obs)
        return_data['logit'] = [1]
        return_data['value'] = [0]

        return return_data

    def process_transition(self, timestep) -> dict:
        step_data = rl_step_data()
        step_data['done'] = timestep.done
        return step_data


class DIStarMockPolicyCollect:

    def __init__(self):

        self.collect_mode = DIstarCollectMode()


def battle_inferencer_for_distar(cfg: EasyDict, env: BaseEnvManager):

    def _battle_inferencer(ctx: "BattleContext"):

        if env.closed:
            env.launch()

        # Get current env obs.
        obs = env.ready_obs
        # {
        #     env_id: {
        #         policy_id:{
        #             'raw_obs': <class 's2clientprotocol.sc2api_pb2.ResponseObservation'>
        #             'opponent_obs': <class 's2clientprotocol.sc2api_pb2.ResponseObservation'>
        #             'action_result': <class 'google.protobuf.pyext._message.RepeatedScalarContainer'>
        #         }
        #     }
        # }

        # the role of remain_episode is to mask necessary rollouts, avoid processing unnecessary data
        # 如果每个 actor 只有一个 env, 下面这部分代码可以全部去掉
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
                # policy.forward
                output = ctx.current_policies[policy_id].forward(policy_obs)
                inference_output[env_id][policy_id] = output
                actions[env_id][policy_id] = output['action']
        #  aciton[env_id][policy_id] = {
        #     'func_id': func_id,
        #     'skip_steps': random.randint(0, MAX_DELAY - 1),
        #     # 'skip_steps': 8,
        #     'queued': random.randint(0, 1),
        #     'unit_tags': unit_tags,
        #     'target_unit_tag': target_unit_tag,
        #     'location': (random.randint(0, SPATIAL_SIZE[0] - 1), random.randint(0, SPATIAL_SIZE[1] - 1))
        # }
        ctx.inference_output = inference_output
        ctx.actions = actions

    return _battle_inferencer


def battle_rolloutor_for_distar(cfg: EasyDict, env: BaseEnvManager, transitions_list: List):

    def _battle_rolloutor(ctx: "BattleContext"):
        timesteps = env.step(ctx.actions)
        #

        ctx.total_envstep_count += len(timesteps)
        ctx.env_step += len(timesteps)
        # for env_id, timestep in timesteps.items():
        # TODO(zms): make sure a standard
        # 这里 timestep 是 一个 env_num 长的 list，但是每次step真的会返回所有 env 的 timestep 吗？（需要确认）是就用 dict，否就用 list
        for env_id, timestep in enumerate(timesteps):
            if timestep.info.get('abnormal'):
                # TODO(zms): cannot get exact env_step of a episode because for each observation,
                # in most cases only one of two policies has a obs.
                # ctx.total_envstep_count -= transitions_list[0].length(env_id)
                # ctx.env_step -= transitions_list[0].length(env_id)

                # TODO(zms): 如果要有available_env_id 的话，这里也要更新
                for transitions in transitions_list:
                    transitions.clear_newest_episode(env_id)
                continue

            for policy_id, _ in enumerate(ctx.current_policies):
                transition = ctx.current_policies[policy_id].process_transition(timestep)
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
