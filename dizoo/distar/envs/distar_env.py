from distar.envs.env import SC2Env

from ding.envs import BaseEnv
from distar.agent.default.lib.actions import ACTIONS, NUM_ACTIONS
from distar.agent.default.lib.features import MAX_DELAY, SPATIAL_SIZE, MAX_SELECTED_UNITS_NUM
from distar.pysc2.lib.action_dict import ACTIONS_STAT
import torch
import random

class DIStarEnv(SC2Env,BaseEnv):

    def __init__(self,cfg):
        super(DIStarEnv, self).__init__(cfg)

    def reset(self):
        return super(DIStarEnv,self).reset()

    def close(self):
        super(DIStarEnv,self).close()

    def step(self,actions):
        # In DI-engine, the return of BaseEnv.step is ('obs', 'reward', 'done', 'info')
        # Here in DI-star, the return is ({'raw_obs': self._obs[agent_idx], 'opponent_obs': opponent_obs, 'action_result': self._action_result[agent_idx]}, reward, episode_complete)
        return super(DIStarEnv,self).step(actions)

    def seed(self, seed, dynamic_seed=False):
        self._random_seed = seed
    
    @property
    def observation_space(self):
        #TODO
        pass

    @property
    def action_space(self):
        #TODO
        pass

    def random_action(self, obs):
        raw = obs[0]['raw_obs'].observation.raw_data

        all_unit_types = set()
        self_unit_types = set()
        
        for u in raw.units:
            # Here we select the units except “buildings that are in building progress” for simplification
            if u.build_progress == 1:
                all_unit_types.add(u.unit_type)
                if u.alliance == 1:
                    self_unit_types.add(u.unit_type)
        
        avail_actions = [
            {0: {'exist_selected_types':[], 'exist_target_types':[]}}, 
            {168:{'exist_selected_types':[], 'exist_target_types':[]}}
        ] # no_op and raw_move_camera don't have seleted_units 

        for action_id, action in ACTIONS_STAT.items():
            exist_selected_types = list(self_unit_types.intersection(set(action['selected_type'])))
            exist_target_types = list(all_unit_types.intersection(set(action['target_type'])))

            # if an action should have target, but we don't have valid target in this observation, then discard this action
            if len(action['target_type']) != 0 and len(exist_target_types) == 0:
                continue

            if len(exist_selected_types) > 0:
                avail_actions.append({action_id: {'exist_selected_types':exist_selected_types, 'exist_target_types':exist_target_types}})
        
        current_action = random.choice(avail_actions)
        func_id, exist_types = current_action.popitem()

        if func_id not in [0, 168]:
            correspond_selected_units = [u.tag for u in raw.units if u.unit_type in exist_types['exist_selected_types'] and u.build_progress == 1]
            correspond_targets = [u.tag for u in raw.units if u.unit_type in exist_types['exist_target_types'] and u.build_progress == 1]

            num_selected_unit = random.randint(0, min(MAX_SELECTED_UNITS_NUM, len(correspond_selected_units)))

            unit_tags =  random.sample(correspond_selected_units, num_selected_unit)
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
            'location': (
                random.randint(0, SPATIAL_SIZE[0] - 1),
                random.randint(0, SPATIAL_SIZE[1] - 1)
            )
        }
        return {0:[data]}


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
