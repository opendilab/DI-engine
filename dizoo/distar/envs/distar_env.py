from ding import policy
from ding.envs import BaseEnv, BaseEnvTimestep

from distar.envs.env import SC2Env
from distar.envs.map_info import get_map_size
from distar.agent.default.lib.features import MAX_DELAY, SPATIAL_SIZE, MAX_SELECTED_UNITS_NUM
from distar.pysc2.lib.action_dict import ACTIONS_STAT

import torch
import random


class DIStarEnv(SC2Env, BaseEnv):

    def __init__(self, cfg):
        super(DIStarEnv, self).__init__(cfg)

    def reset(self):
        observations, game_info, map_name = super(DIStarEnv, self).reset()
        map_size = get_map_size(map_name)

        for policy_id, policy_obs in observations.items():
            policy_obs['game_info'] = game_info[policy_id]
            policy_obs['map_name'] = map_name
            policy_obs['map_size'] = map_size

        return observations

    def close(self):
        super(DIStarEnv, self).close()

    def step(self, actions):
        # In DI-engine, the return of BaseEnv.step is ('obs', 'reward', 'done', 'info')
        # Here in DI-star, the return is ({'raw_obs': self._obs[agent_idx], 'opponent_obs': opponent_obs, 'action_result': self._action_result[agent_idx]}, reward, episode_complete)
        next_observations, reward, done = super(DIStarEnv, self).step(actions)
        # next_observations 和 observations 格式一样
        # reward 是 list [policy reward 1, policy reward 2]
        # done 是 一个 bool 值
        info = {}
        for policy_id in range(self._num_agents):
            info[policy_id] = {}
            if done:
                info[policy_id]['final_eval_reward'] = reward[policy_id]
        timestep = BaseEnvTimestep(obs=next_observations, reward=reward, done=done, info=info)
        return timestep

    def seed(self, seed, dynamic_seed=False):
        self._random_seed = seed

    @property
    def game_info(self):
        return self._game_info

    @property
    def map_name(self):
        return self._map_name

    @property
    def observation_space(self):
        #TODO
        pass

    @property
    def action_space(self):
        #TODO
        pass

    @classmethod
    def random_action(cls, obs):
        raw = obs['raw_obs'].observation.raw_data

        all_unit_types = set()
        self_unit_types = set()

        for u in raw.units:
            # Here we select the units except “buildings that are in building progress” for simplification
            if u.build_progress == 1:
                all_unit_types.add(u.unit_type)
                if u.alliance == 1:
                    self_unit_types.add(u.unit_type)

        avail_actions = [
            {
                0: {
                    'exist_selected_types': [],
                    'exist_target_types': []
                }
            }, {
                168: {
                    'exist_selected_types': [],
                    'exist_target_types': []
                }
            }
        ]  # no_op and raw_move_camera don't have seleted_units

        for action_id, action in ACTIONS_STAT.items():
            exist_selected_types = list(self_unit_types.intersection(set(action['selected_type'])))
            exist_target_types = list(all_unit_types.intersection(set(action['target_type'])))

            # if an action should have target, but we don't have valid target in this observation, then discard this action
            if len(action['target_type']) != 0 and len(exist_target_types) == 0:
                continue

            if len(exist_selected_types) > 0:
                avail_actions.append(
                    {
                        action_id: {
                            'exist_selected_types': exist_selected_types,
                            'exist_target_types': exist_target_types
                        }
                    }
                )

        current_action = random.choice(avail_actions)
        func_id, exist_types = current_action.popitem()

        if func_id not in [0, 168]:
            correspond_selected_units = [
                u.tag for u in raw.units if u.unit_type in exist_types['exist_selected_types'] and u.build_progress == 1
            ]
            correspond_targets = [
                u.tag for u in raw.units if u.unit_type in exist_types['exist_target_types'] and u.build_progress == 1
            ]

            num_selected_unit = random.randint(0, min(MAX_SELECTED_UNITS_NUM, len(correspond_selected_units)))

            unit_tags = random.sample(correspond_selected_units, num_selected_unit)
            target_unit_tag = random.choice(correspond_targets) if len(correspond_targets) > 0 else None

        else:
            unit_tags = []
            target_unit_tag = None

        data = {
            'func_id': func_id,
            'skip_steps': random.randint(0, MAX_DELAY - 1),
            # 'skip_steps': 8,
            'queued': random.randint(0, 1),
            'unit_tags': unit_tags,
            'target_unit_tag': target_unit_tag,
            'location': (random.randint(0, SPATIAL_SIZE[0] - 1), random.randint(0, SPATIAL_SIZE[1] - 1))
        }
        return [data]

    @property
    def reward_space(self):

        #TODO
        pass

    def __repr__(self):
        return "DI-engine DI-star Env"


# if __name__ == '__main__':
#     no_target_unit_actions = sorted([action['func_id'] for action in ACTIONS if action['target_unit'] == False])
#     no_target_unit_actions_dict = sorted([action_id for action_id, action in ACTIONS_STAT.items() if len(action['target_type']) == 0])
#     print(no_target_unit_actions)
#     print(no_target_unit_actions_dict)
#     assert no_target_unit_actions == no_target_unit_actions_dict
