from namedlist import namedlist
import numpy as np
import gym
from typing import Any, Union, List
import copy

from overcooked_ai_py.mdp.actions import Action, Direction
from overcooked_ai_py.mdp.overcooked_mdp import PlayerState, OvercookedGridworld, OvercookedState, ObjectState, SoupState, Recipe
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv, DEFAULT_ENV_PARAMS

from ding.envs import BaseEnv, BaseEnvTimestep, BaseEnvInfo
from ding.envs.common.env_element import EnvElement, EnvElementInfo
from ding.utils import ENV_REGISTRY

OvercookEnvTimestep = namedlist('OvercookEnvTimestep', ['obs', 'reward', 'done', 'info'])
OvercookEnvInfo = namedlist('OvercookEnvInfo', ['agent_num', 'obs_space', 'act_space', 'rew_space'])

# n, s = Direction.NORTH, Direction.SOUTH
# e, w = Direction.EAST, Direction.WEST
# stay, interact = Action.STAY, Action.INTERACT
# Action.ALL_ACTIONS: [n, s, e, w, stay, interact]


@ENV_REGISTRY.register('overcooked')
class OvercookEnv(BaseEnv):

    def __init__(self, cfg) -> None:
        self._cfg = cfg
        self._env_name = cfg.get("env_name", "cramped_room")
        self._horizon = cfg.get("horizon", 400)
        self._concat_obs = cfg.get("concat_obs", False)
        self._action_mask = cfg.get("action_mask", True)
        self._use_shaped_reward = cfg.get("use_shaped_reward", True)
        self.mdp = OvercookedGridworld.from_layout_name(self._env_name)
        self.base_env = OvercookedEnv.from_mdp(self.mdp, horizon=self._horizon, info_level=0)
        featurize_fn = lambda mdp, state: mdp.lossless_state_encoding(state)
        self.featurize_fn = featurize_fn
        self.action_dim = len(Action.ALL_ACTIONS)
        self.action_space = gym.spaces.Discrete(len(Action.ALL_ACTIONS))
        # rightnow overcook environment encoding only support 2 agent game
        self.agent_num = 2
        # set up obs shape
        dummy_mdp = self.base_env.mdp
        dummy_state = dummy_mdp.get_standard_start_state()
        self.obs_shape = self.featurize_fn(dummy_mdp, dummy_state)[0].shape

    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        self._seed = seed
        self._dynamic_seed = dynamic_seed
        np.random.seed(self._seed)

    def close(self) -> None:
        # Note: the real env instance only has a empty close method, only pas
        pass

    def step(self, action):
        if isinstance(action, list):
            action = np.concatenate(action)
        assert all(self.action_space.contains(a) for a in action), "%r (%s) invalid" % (action, type(action))
        agent_action, other_agent_action = [Action.INDEX_TO_ACTION[a] for a in action]

        if self.agent_idx == 0:
            joint_action = (agent_action, other_agent_action)
        else:
            joint_action = (other_agent_action, agent_action)

        next_state, reward, done, env_info = self.base_env.step(joint_action)
        if self._use_shaped_reward:
            reward += env_info['shaped_r_by_agent'][0]
            reward += env_info['shaped_r_by_agent'][1]

        reward = np.array([float(reward)])
        self._final_eval_reward += reward
        ob_p0, ob_p1 = self.featurize_fn(self.mdp, next_state)
        if self.agent_idx == 0:
            both_agents_ob = [ob_p0, ob_p1]
        else:
            both_agents_ob = [ob_p1, ob_p0]
        if self._concat_obs:
            both_agents_ob = np.concatenate(both_agents_ob)
        else:
            both_agents_ob = np.stack(both_agents_ob)

        env_info["policy_agent_idx"] = self.agent_idx
        env_info["final_eval_reward"] = self._final_eval_reward

        action_mask = self.get_action_mask()
        if self._action_mask:
            obs = {
                "agent_state": both_agents_ob,
                "overcooked_state": self.base_env.state,
                "other_agent_env_idx": 1 - self.agent_idx,
                "action_mask": action_mask
            }
        else:
            obs = both_agents_ob
        return OvercookEnvTimestep(obs, reward, done, env_info)

    def reset(self):
        self.base_env.reset()
        self._final_eval_reward = 0
        self.mdp = self.base_env.mdp
        # random init agent index
        self.agent_idx = np.random.choice([0, 1])
        ob_p0, ob_p1 = self.featurize_fn(self.mdp, self.base_env.state)

        if self.agent_idx == 0:
            both_agents_ob = [ob_p0, ob_p1]
        else:
            both_agents_ob = [ob_p1, ob_p0]
        if self._concat_obs:
            both_agents_ob = np.concatenate(both_agents_ob)
        else:
            both_agents_ob = np.stack(both_agents_ob)

        action_mask = self.get_action_mask()

        if self._action_mask:
            obs = {
                "agent_state": both_agents_ob,
                "overcooked_state": self.base_env.state,
                "other_agent_env_idx": 1 - self.agent_idx,
                "action_mask": action_mask
            }
        else:
            obs = both_agents_ob
        return obs

    def get_available_actions(self):
        return self.mdp.get_actions(self.base_env.state)

    def get_action_mask(self):
        available_actions = self.get_available_actions()

        action_masks = np.zeros((2, self.action_dim))

        for i in range(self.action_dim):
            if Action.INDEX_TO_ACTION[i] in available_actions[0]:
                action_masks[0][i] = 1
            if Action.INDEX_TO_ACTION[i] in available_actions[1]:
                action_masks[1][i] = 1

        return action_masks

    def info(self):
        T = EnvElementInfo
        if self._concat_obs:
            agent_state = list(self.obs_shape)
            agent_state[0] = agent_state[0] * 2
            agent_state = tuple(agent_state)
        else:
            agent_state = (self.agent_num, self.obs_shape)
        env_info = OvercookEnvInfo(
            agent_num=self.agent_num,
            obs_space=T({
                'agent_state': agent_state,
                'action_mask': (self.agent_num, self.action_dim),
            }, None),
            act_space=T((self.agent_num, self.action_dim), None),
            rew_space=T((1, ), None)
        )
        return env_info

    def __repr__(self):
        pass


@ENV_REGISTRY.register('overcooked_game')
class OvercookGameEnv(BaseEnv):

    def __init__(self, cfg) -> None:
        self._cfg = cfg
        self._env_name = cfg.get("env_name", "cramped_room")
        self._horizon = cfg.get("horizon", 400)
        self._concat_obs = cfg.get("concat_obs", False)
        self._action_mask = cfg.get("action_mask", False)
        self._use_shaped_reward = cfg.get("use_shaped_reward", True)
        self.mdp = OvercookedGridworld.from_layout_name(self._env_name)
        self.base_env = OvercookedEnv.from_mdp(self.mdp, horizon=self._horizon, info_level=0)
        featurize_fn = lambda mdp, state: mdp.lossless_state_encoding(state)
        self.featurize_fn = featurize_fn
        self.action_dim = len(Action.ALL_ACTIONS)
        self.action_space = gym.spaces.Discrete(len(Action.ALL_ACTIONS))
        # rightnow overcook environment encoding only support 2 agent game
        self.agent_num = 2
        # set up obs shape
        dummy_mdp = self.base_env.mdp
        dummy_state = dummy_mdp.get_standard_start_state()
        self.obs_shape = self.featurize_fn(dummy_mdp, dummy_state)[0].shape

    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        self._seed = seed
        self._dynamic_seed = dynamic_seed
        np.random.seed(self._seed)

    def close(self) -> None:
        # Note: the real env instance only has a empty close method, only pas
        pass

    def step(self, action):
        if isinstance(action, list):
            action = np.array(action).astype(np.int)
            if action.shape == (2, 1):
                action = [action[0][0], action[1][0]]
        assert all(self.action_space.contains(a) for a in action), "%r (%s) invalid" % (action, type(action))
        agent_action, other_agent_action = [Action.INDEX_TO_ACTION[a] for a in action]

        if self.agent_idx == 0:
            joint_action = (agent_action, other_agent_action)
        else:
            joint_action = (other_agent_action, agent_action)

        next_state, reward, done, env_info = self.base_env.step(joint_action)

        reward = np.array([float(reward)])
        self._final_eval_reward += reward
        if self._use_shaped_reward:
            self._final_eval_reward += env_info['shaped_r_by_agent'][0]
            self._final_eval_reward += env_info['shaped_r_by_agent'][1]
        rewards = np.array([reward, reward]).astype(np.float32)
        if self._use_shaped_reward:
            rewards[0] += env_info['shaped_r_by_agent'][0]
            rewards[1] += env_info['shaped_r_by_agent'][1]
        ob_p0, ob_p1 = self.featurize_fn(self.mdp, next_state)
        if self.agent_idx == 0:
            both_agents_ob = [ob_p0, ob_p1]
        else:
            both_agents_ob = [ob_p1, ob_p0]
        if self._concat_obs:
            both_agents_ob = np.concatenate(both_agents_ob)
        else:
            both_agents_ob = np.stack(both_agents_ob)

        env_info["policy_agent_idx"] = self.agent_idx
        env_info["final_eval_reward"] = self._final_eval_reward

        action_mask = self.get_action_mask()
        if self._action_mask:
            obs = {
                "agent_state": both_agents_ob,
                "overcooked_state": self.base_env.state,
                "other_agent_env_idx": 1 - self.agent_idx,
                "action_mask": action_mask
            }
        else:
            obs = both_agents_ob
        return OvercookEnvTimestep(obs, rewards, done, [env_info, env_info])

    def reset(self):
        self.base_env.reset()
        self._final_eval_reward = 0
        self.mdp = self.base_env.mdp
        # random init agent index
        self.agent_idx = np.random.choice([0, 1])
        #fix init agent index
        self.agent_idx = 0
        ob_p0, ob_p1 = self.featurize_fn(self.mdp, self.base_env.state)

        if self.agent_idx == 0:
            both_agents_ob = [ob_p0, ob_p1]
        else:
            both_agents_ob = [ob_p1, ob_p0]
        if self._concat_obs:
            both_agents_ob = np.concatenate(both_agents_ob)
        else:
            both_agents_ob = np.stack(both_agents_ob)

        action_mask = self.get_action_mask()

        if self._action_mask:
            obs = {
                "agent_state": both_agents_ob,
                "overcooked_state": self.base_env.state,
                "other_agent_env_idx": 1 - self.agent_idx,
                "action_mask": action_mask
            }
        else:
            obs = both_agents_ob
        return obs

    def get_available_actions(self):
        return self.mdp.get_actions(self.base_env.state)

    def get_action_mask(self):
        available_actions = self.get_available_actions()

        action_masks = np.zeros((2, self.action_dim))

        for i in range(self.action_dim):
            if Action.INDEX_TO_ACTION[i] in available_actions[0]:
                action_masks[0][i] = 1
            if Action.INDEX_TO_ACTION[i] in available_actions[1]:
                action_masks[1][i] = 1

        return action_masks

    def info(self):
        T = EnvElementInfo
        if self._concat_obs:
            agent_state = list(self.obs_shape)
            agent_state[0] = agent_state[0] * 2
            agent_state = tuple(agent_state)
        else:
            agent_state = (self.agent_num, self.obs_shape)
        env_info = OvercookEnvInfo(
            agent_num=self.agent_num,
            obs_space=T({
                'agent_state': agent_state,
                'action_mask': (self.agent_num, self.action_dim),
            }, None),
            act_space=T((self.agent_num, self.action_dim), None),
            rew_space=T((1, ), None)
        )
        return env_info

    def __repr__(self):
        return "DI-engine Overcooked GameEnv"