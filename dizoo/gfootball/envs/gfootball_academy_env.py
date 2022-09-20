"""
The code below is adapted from https://github.com/lich14/CDS/tree/main/CDS_GRF/envs/grf,
which is from the codebase of the CDS paper "Celebrating Diversity in Shared Multi-Agent Reinforcement Learning"
"""

import gfootball.env as football_env
from gfootball.env import observation_preprocessing
import gym
import numpy as np
from ding.utils import ENV_REGISTRY
from typing import Any, List, Union, Optional
import copy
import torch
from ding.envs import BaseEnv, BaseEnvTimestep
from ding.torch_utils import to_ndarray, to_list
import os
from matplotlib import animation
import matplotlib.pyplot as plt


@ENV_REGISTRY.register('gfootball-academy')
class GfootballAcademyEnv(BaseEnv):

    def __init__(
        self,
        cfg: dict,
        dense_reward=False,
        write_full_episode_dumps=False,
        write_goal_dumps=False,
        dump_freq=1000,
        render=False,
        time_limit=150,
        time_step=0,
        stacked=False,
        representation="simple115",
        rewards='scoring',
        logdir='football_dumps',
        write_video=True,
        number_of_right_players_agent_controls=0,
    ):
        """
        'academy_3_vs_1_with_keeper'
        n_agents=3,
        obs_dim=26,
        'academy_counterattack_hard'
        n_agents=4,
        obs_dim=34,
        """
        self._cfg = cfg
        self._save_replay = False
        self._save_replay_count = 0
        self._replay_path = None
        self.dense_reward = dense_reward
        self.write_full_episode_dumps = write_full_episode_dumps
        self.write_goal_dumps = write_goal_dumps
        self.dump_freq = dump_freq
        self.render = render
        self.env_name = self._cfg.env_name  # TODO
        self.n_agents = self._cfg.agent_num
        self.obs_dim = self._cfg.obs_dim

        self.episode_limit = time_limit
        self.time_step = time_step
        self.stacked = stacked
        self.representation = representation
        self.rewards = rewards
        self.logdir = logdir
        self.write_video = write_video
        self.number_of_right_players_agent_controls = number_of_right_players_agent_controls

        self._env = football_env.create_environment(
            write_full_episode_dumps=self.write_full_episode_dumps,
            write_goal_dumps=self.write_goal_dumps,
            env_name=self.env_name,
            stacked=self.stacked,
            representation=self.representation,
            rewards=self.rewards,
            logdir=self.logdir,
            render=self.render,
            write_video=self.write_video,
            dump_frequency=self.dump_freq,
            number_of_left_players_agent_controls=self.n_agents,
            number_of_right_players_agent_controls=self.number_of_right_players_agent_controls,
            channel_dimensions=(observation_preprocessing.SMM_WIDTH, observation_preprocessing.SMM_HEIGHT)
        )

        obs_space_low = self._env.observation_space.low[0][:self.obs_dim]
        obs_space_high = self._env.observation_space.high[0][:self.obs_dim]

        self._action_space = gym.spaces.Dict(
            {agent_i: gym.spaces.Discrete(self._env.action_space.nvec[1])
             for agent_i in range(self.n_agents)}
        )
        self._observation_space = gym.spaces.Dict(
            {
                agent_i:
                gym.spaces.Box(low=obs_space_low, high=obs_space_high, dtype=self._env.observation_space.dtype)
                for agent_i in range(self.n_agents)
            }
        )
        self._reward_space = gym.spaces.Box(low=0, high=100, shape=(1, ), dtype=np.float32)  # TODO(pu)

        self.n_actions = self.action_space[0].n

    def get_simple_obs(self, index=-1):
        full_obs = self._env.unwrapped.observation()[0]
        simple_obs = []

        if self.env_name == 'academy_3_vs_1_with_keeper':
            if index == -1:
                # global state, absolute position
                simple_obs.append(full_obs['left_team'][-self.n_agents:].reshape(-1))
                simple_obs.append(full_obs['left_team_direction'][-self.n_agents:].reshape(-1))

                simple_obs.append(full_obs['right_team'].reshape(-1))
                simple_obs.append(full_obs['right_team_direction'].reshape(-1))

                simple_obs.append(full_obs['ball'])
                simple_obs.append(full_obs['ball_direction'])
            else:
                # local state, relative position
                ego_position = full_obs['left_team'][-self.n_agents + index].reshape(-1)
                simple_obs.append(ego_position)
                simple_obs.append(
                    (np.delete(full_obs['left_team'][-self.n_agents:], index, axis=0) - ego_position).reshape(-1)
                )

                simple_obs.append(full_obs['left_team_direction'][-self.n_agents + index].reshape(-1))
                simple_obs.append(
                    np.delete(full_obs['left_team_direction'][-self.n_agents:], index, axis=0).reshape(-1)
                )

                simple_obs.append((full_obs['right_team'] - ego_position).reshape(-1))
                simple_obs.append(full_obs['right_team_direction'].reshape(-1))

                simple_obs.append(full_obs['ball'][:2] - ego_position)
                simple_obs.append(full_obs['ball'][-1].reshape(-1))
                simple_obs.append(full_obs['ball_direction'])

        elif self.env_name == 'academy_counterattack_hard':
            if index == -1:
                # global state, absolute position
                simple_obs.append(full_obs['left_team'][-self.n_agents:].reshape(-1))
                simple_obs.append(full_obs['left_team_direction'][-self.n_agents:].reshape(-1))

                simple_obs.append(full_obs['right_team'][0])
                simple_obs.append(full_obs['right_team'][1])
                simple_obs.append(full_obs['right_team'][2])
                simple_obs.append(full_obs['right_team_direction'][0])
                simple_obs.append(full_obs['right_team_direction'][1])
                simple_obs.append(full_obs['right_team_direction'][2])

                simple_obs.append(full_obs['ball'])
                simple_obs.append(full_obs['ball_direction'])

            else:
                # local state, relative position
                ego_position = full_obs['left_team'][-self.n_agents + index].reshape(-1)
                simple_obs.append(ego_position)
                simple_obs.append(
                    (np.delete(full_obs['left_team'][-self.n_agents:], index, axis=0) - ego_position).reshape(-1)
                )

                simple_obs.append(full_obs['left_team_direction'][-self.n_agents + index].reshape(-1))
                simple_obs.append(
                    np.delete(full_obs['left_team_direction'][-self.n_agents:], index, axis=0).reshape(-1)
                )

                simple_obs.append(full_obs['right_team'][0] - ego_position)
                simple_obs.append(full_obs['right_team'][1] - ego_position)
                simple_obs.append(full_obs['right_team'][2] - ego_position)
                simple_obs.append(full_obs['right_team_direction'][0])
                simple_obs.append(full_obs['right_team_direction'][1])
                simple_obs.append(full_obs['right_team_direction'][2])

                simple_obs.append(full_obs['ball'][:2] - ego_position)
                simple_obs.append(full_obs['ball'][-1].reshape(-1))
                simple_obs.append(full_obs['ball_direction'])

        simple_obs = np.concatenate(simple_obs)
        return simple_obs

    def get_global_state(self):
        return self.get_simple_obs(-1)

    def get_global_special_state(self):
        return [np.concatenate([self.get_global_state(), self.get_obs_agent(i)]) for i in range(self.n_agents)]

    def check_if_done(self):
        cur_obs = self._env.unwrapped.observation()[0]
        ball_loc = cur_obs['ball']
        ours_loc = cur_obs['left_team'][-self.n_agents:]

        if ball_loc[0] < 0 or any(ours_loc[:, 0] < 0):
            """
            This is based on the CDS paper:
            'We make a small and reasonable change to the half-court offensive scenarios: our players will lose if
            they or the ball returns to our half-court.'
            """
            return True

        return False

    def reset(self):
        """Returns initial observations and states."""
        if self._save_replay:
            self._frames = []
        self.time_step = 0
        self._env.reset()
        obs = {
            'agent_state': np.stack(self.get_obs(), axis=0).astype(np.float32),
            # Note: here 'global_state' is the agent_specific_global_state,
            # we simply concatenate the global_state and agent_state
            'global_state': np.stack(
                self.get_global_special_state(),
                axis=0,
            ).astype(np.float32),
            'action_mask': np.stack(self.get_avail_actions(), axis=0).astype(np.float32),
        }

        if hasattr(self, '_seed') and hasattr(self, '_dynamic_seed') and self._dynamic_seed:
            np_seed = 100 * np.random.randint(1, 1000)
            self._env.seed(self._seed + np_seed)
        elif hasattr(self, '_seed'):
            self._env.seed(self._seed)
        self._final_eval_reward = 0

        return obs

    def step(self, actions):
        """Returns reward, terminated, info."""
        assert isinstance(actions, np.ndarray) or isinstance(actions, list), type(actions)
        self.time_step += 1
        if isinstance(actions, np.ndarray):
            actions = actions.tolist()

        if self._save_replay:
            self._frames.append(self._env.render(mode='rgb_array'))
        _, original_rewards, done, infos = self._env.step(actions)
        obs = {
            'agent_state': np.stack(self.get_obs(), axis=0).astype(np.float32),
            # Note: here 'global_state' is the agent_specific_global_state,
            # we simply concatenate the global_state and agent_state
            'global_state': np.stack(
                self.get_global_special_state(),
                axis=0,
            ).astype(np.float32),
            'action_mask': np.stack(self.get_avail_actions(), axis=0).astype(np.float32),
        }
        rewards = list(original_rewards)

        if self.time_step >= self.episode_limit:
            done = True

        if self.check_if_done():
            done = True

        if done:
            if self._save_replay:
                path = os.path.join(
                    self._replay_path, '{}_episode_{}.gif'.format(self.env_name, self._save_replay_count)
                )
                self.display_frames_as_gif(self._frames, path)
                self._save_replay_count += 1

        if sum(rewards) <= 0:
            """
            This is based on the CDS paper:
            "Environmental reward only occurs at the end of the game.
            They will get +100 if they win, else get -1."
            If done=False, the reward is -1,
            If done=True and sum(rewards)<=0 the reward is 1.
            If done=True and sum(rewards)>0 the reward is 100.
            """
            infos['final_eval_reward'] = infos['score_reward']  # TODO(pu)
            return BaseEnvTimestep(obs, np.array(-int(done)).astype(np.float32), done, infos)
        else:
            infos['final_eval_reward'] = infos['score_reward']
            return BaseEnvTimestep(obs, np.array(100).astype(np.float32), done, infos)

    def get_obs(self):
        """Returns all agent observations in a list."""
        obs = [self.get_simple_obs(i) for i in range(self.n_agents)]
        return obs

    def get_obs_agent(self, agent_id):
        """Returns observation for agent_id."""
        return self.get_simple_obs(agent_id)

    def get_obs_size(self):
        """Returns the size of the observation."""
        return self.obs_dim

    def get_state(self):
        """Returns the global state."""
        return self.get_global_state()

    def get_state_size(self):
        """Returns the size of the global state."""
        return self.obs_dim

    def get_avail_actions(self):
        """Returns the available actions of all agents in a list."""
        return [[1 for _ in range(self.n_actions)] for agent_id in range(self.n_agents)]

    def get_avail_agent_actions(self, agent_id):
        """Returns the available actions for agent_id."""
        return self.get_avail_actions()[agent_id]

    def render(self):
        pass

    def close(self):
        self._env.close()

    def save_replay(self):
        """Save a replay."""
        pass

    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        self._seed = seed
        self._dynamic_seed = dynamic_seed
        np.random.seed(self._seed)

    def random_action(self) -> np.ndarray:
        random_action = self.action_space.sample()
        random_action = to_ndarray([random_action], dtype=np.int64)
        return random_action

    @property
    def observation_space(self) -> gym.spaces.Space:
        return self._observation_space

    @property
    def action_space(self) -> gym.spaces.Space:
        return self._action_space

    @property
    def reward_space(self) -> gym.spaces.Space:
        return self._reward_space

    def __repr__(self) -> str:
        return f'GfootballEnv Academy Env {self.env_name}'

    def enable_save_replay(self, replay_path: Optional[str] = None) -> None:
        """
        Overview:
            Save replay file in the given path
        Arguments:
            - replay_path(:obj:`str`): Storage path.
        """
        if replay_path is None:
            replay_path = './video'
        self._save_replay = True
        self._replay_path = replay_path
        self._save_replay_count = 0

    @staticmethod
    def display_frames_as_gif(frames: list, path: str) -> None:
        patch = plt.imshow(frames[0])
        plt.axis('off')

        def animate(i):
            patch.set_data(frames[i])

        anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=5)
        anim.save(path, writer='imagemagick', fps=20)
