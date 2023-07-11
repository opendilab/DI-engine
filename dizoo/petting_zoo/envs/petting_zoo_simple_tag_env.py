from typing import Any, List, Union, Optional, Dict
import gymnasium as gym
import numpy as np
import pettingzoo
from functools import reduce

from ding.envs import BaseEnv, BaseEnvTimestep, FrameStackWrapper
from ding.torch_utils import to_ndarray, to_list
from ding.envs.common.common_function import affine_transform
from ding.utils import ENV_REGISTRY, import_module
from pettingzoo.utils.conversions import parallel_wrapper_fn
from pettingzoo.mpe._mpe_utils.simple_env import SimpleEnv, make_env
from pettingzoo.mpe.simple_tag.simple_tag import Scenario


@ENV_REGISTRY.register('petting_zoo_tag')
class PettingZooTagEnv(BaseEnv):
    # Now only supports simple_spread_v2.
    # All agents' observations should have the same shape.

    def __init__(self, cfg: dict) -> None:
        self._cfg = cfg
        self._init_flag = False
        self._replay_path = None
        self._env_family = self._cfg.env_family
        self._env_id = self._cfg.env_id
        # self._num_agents = self._cfg.n_agent
        # self._num_landmarks = self._cfg.n_landmark
        self._num_good = self._cfg.num_good
        self._num_adversaries = self._cfg.num_adversaries
        self._num_obstacles = self._cfg.num_obstacles
        self._continuous_actions = self._cfg.get('continuous_actions', False)
        self._max_cycles = self._cfg.get('max_cycles', 25)
        self._act_scale = self._cfg.get('act_scale', False)
        self._agent_specific_global_state = self._cfg.get('agent_specific_global_state', False)
        if self._act_scale:
            assert self._continuous_actions, 'Only continuous action space env needs act_scale'

    def reset(self) -> np.ndarray:
        if not self._init_flag:
            # In order to align with the simple spread in Multiagent Particle Env (MPE),
            # instead of adopting the pettingzoo interface directly,
            # we have redefined the way rewards are calculated

            # import_module(['pettingzoo.{}.{}'.format(self._env_family, self._env_id)])
            # self._env = pettingzoo.__dict__[self._env_family].__dict__[self._env_id].parallel_env(
            #     N=self._cfg.n_agent, continuous_actions=self._continuous_actions, max_cycles=self._max_cycles
            # )

            # init parallel_env wrapper
            _env = make_env(simple_tag_raw_env)
            parallel_env = parallel_wrapper_fn(_env)
            # init env
            self._env = parallel_env(
                num_good=self._cfg.num_good,
                num_adversaries=self._cfg.num_adversaries,
                num_obstacles=self._cfg.num_obstacles,
                continuous_actions=self._continuous_actions,
                max_cycles=self._max_cycles
            )
        # dynamic seed reduces training speed greatly
        # if hasattr(self, '_seed') and hasattr(self, '_dynamic_seed') and self._dynamic_seed:
        #     np_seed = 100 * np.random.randint(1, 1000)
        #     self._env.seed(self._seed + np_seed)
        if self._replay_path is not None:
            self._env = gym.wrappers.Monitor(
                self._env, self._replay_path, video_callable=lambda episode_id: True, force=True
            )
        if hasattr(self, '_seed'):
            obs = self._env.reset(seed=self._seed)
        else:
            obs = self._env.reset()
        if not self._init_flag:
            self._agents = self._env.agents

            self._action_space = gym.spaces.Dict({agent: self._env.action_space(agent) for agent in self._agents})
            single_agent_obs_space = self._env.action_space(self._agents[0])
            # single_agent_obs_space = gym.spaces.Dict({agent: self._env.observation_space(agent) for agent in self._agents})
            # here
            if isinstance(single_agent_obs_space, gym.spaces.Box):
                self._action_dim = single_agent_obs_space.shape
            elif isinstance(single_agent_obs_space, gym.spaces.Discrete):
                self._action_dim = (single_agent_obs_space.n, )
            else:
                raise Exception('Only support `Box` or `Discrete` obs space for single agent.')

            # only for env 'simple_spread_v2', n_agent = 5
            # now only for the case that each agent in the team have the same obs structure and corresponding shape.
            if not self._cfg.agent_obs_only:
                self._observation_space = gym.spaces.Dict(
                    {
                        # 'agent_state': gym.spaces.Box(
                        #     low=float("-inf"),
                        #     high=float("inf"),
                        #     shape=(self._num_agents,
                        #            self._env.observation_space('agent_0').shape[0]),  # (self._num_agents, 30)
                        #     dtype=np.float32
                        # ),
                        'agent_state':gym.spaces.Dict({agent: self._env.observation_space(agent) for agent in self._agents}),
                        'global_state': gym.spaces.Box(
                            low=float("-inf"),
                            high=float("inf"),
                            shape=(
                                (2+2)*(self._num_good+self._num_adversaries)+2*self._num_obstacles,
                            ),
                            dtype=np.float32
                        ),
                        'agent_alone_state': gym.spaces.Box(
                            low=float("-inf"),
                            high=float("inf"),
                            shape=(self._num_good+self._num_adversaries, 2+2+2*self._num_obstacles),
                            dtype=np.float32
                        ),
                        'agent_alone_padding_state': gym.spaces.Dict({agent: self._env.observation_space(agent) for agent in self._agents}),
                        'action_mask': gym.spaces.Box(
                            low=float("-inf"),
                            high=float("inf"),
                            shape=(self._num_good+self._num_adversaries, self._action_dim[0]),  # (self._num_agents, 5)
                            dtype=np.float32
                        )
                    }
                )
                # whether use agent_specific_global_state. It is usually used in AC multiagent algos, e.g., mappo, masac, etc.
                if self._agent_specific_global_state:
                    agent_specifig_global_state = gym.spaces.Dict(
                        {agent: gym.spaces.Box(
                            low=float("-inf"),
                            high=float("inf"),
                            shape=(self._env.observation_space(agent).shape[0]+(2+2)*(self._num_good+self._num_adversaries)+2*self._num_obstacles,),
                            dtype=np.float32
                        )
                        for agent in self._agents})
                    self._observation_space['global_state'] = agent_specifig_global_state
            else:
                # for case when env.agent_obs_only=True
                self._observation_space = gym.spaces.Dict({agent: self._env.observation_space(agent) for agent in self._agents})

            self._reward_space = gym.spaces.Dict(
                {
                    agent: gym.spaces.Box(low=float("-inf"), high=float("inf"), shape=(1, ), dtype=np.float32)
                    for agent in self._agents
                }
            )
            self._init_flag = True
        # self._eval_episode_return = {agent: 0. for agent in self._agents}
        self._eval_episode_return = 0.
        self._step_count = 0
        obs_n = self._process_obs(obs)
        return obs_n

    def close(self) -> None:
        if self._init_flag:
            self._env.close()
        self._init_flag = False

    def render(self) -> None:
        self._env.render()

    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        self._seed = seed
        self._dynamic_seed = dynamic_seed
        np.random.seed(self._seed)

    def step(self, action: np.ndarray) -> BaseEnvTimestep:
        self._step_count += 1
        assert isinstance(action, np.ndarray), type(action)
        action = self._process_action(action)
        if self._act_scale:
            for agent in self._agents:
                # print(action[agent])
                # print(self.action_space[agent])
                # print(self.action_space[agent].low, self.action_space[agent].high)
                action[agent] = affine_transform(
                    action[agent], min_val=self.action_space[agent].low, max_val=self.action_space[agent].high
                )

        obs, rew, done, trunc, info = self._env.step(action)
        obs_n = self._process_obs(obs)
        rew_n = np.array([sum([rew[agent] for agent in self._agents])])
        rew_n = rew_n.astype(np.float32)
        # collide_sum = 0
        # for i in range(self._num_agents):
        #     collide_sum += info['n'][i][1]
        # collide_penalty = self._cfg.get('collide_penal', self._num_agent)
        # rew_n += collide_sum * (1.0 - collide_penalty)
        # rew_n = rew_n / (self._cfg.get('max_cycles', 25) * self._num_agent)
        self._eval_episode_return += rew_n.item()

        # occupied_landmarks = info['n'][0][3]
        # if self._step_count >= self._max_step or occupied_landmarks >= self._n_agent \
        #         or occupied_landmarks >= self._num_landmarks:
        #     done_n = True
        # else:
        #     done_n = False
        done_n = reduce(lambda x, y: x and y, done.values()) or self._step_count >= self._max_cycles

        # for agent in self._agents:
        #     self._eval_episode_return[agent] += rew[agent]
        if done_n:  # or reduce(lambda x, y: x and y, done.values())
            info['eval_episode_return'] = self._eval_episode_return
        # for agent in rew:
        #     rew[agent] = to_ndarray([rew[agent]])
        return BaseEnvTimestep(obs_n, rew_n, done_n, info)

    def enable_save_replay(self, replay_path: Optional[str] = None) -> None:
        if replay_path is None:
            replay_path = './video'
        self._replay_path = replay_path

    def _process_obs(self, obs: 'torch.Tensor') -> np.ndarray:  # noqa
        obs = obs
        if self._cfg.get('agent_obs_only', False):
            return obs
        ret = {}
        # Raw agent observation structure is --
        # [self_vel, self_pos, landmark_rel_positions, other_agent_rel_positions, other_agent_velocities]
        # where `other_agent_velocities` are v of other good agents

        # agent_state: Dict like {'adversary0':xx, 'agent':xx}. 
        #              Shape:
        #              - good_agent: 2+2+2*num_obstacles+2*(num_good+num_adversaries-1)+2*(num_good-1)
        #              - adversary: 2+2+2*num_obstacles+2*(num_good+num_adversaries-1)+2*num_good
        #              Stacked observation. Contains
        #              - agent itself's state(velocity + position)
        #              - position of obstacles and other agents
        #              - other good agent velocities
        ret['agent_state'] = obs
        # global_state: Shape: (2+2)*(num_good+num_adversaries)+2*num_obstacles.
        #               1-dim vector. Contains
        #               - all good agents' state(velocity + position) +
        #               - all adversaries' state(velocity + position) +
        #               - all obstacles' position.
        all_agent_v = np.concatenate([value[0:2] for value in obs.values()])
        all_agent_pos = np.concatenate([value[2:4] for value in obs.values()])
        all_obstacle_pos = list(obs.values())[0][4:4+self._num_obstacles*2]
        ret['global_state'] = np.concatenate([all_agent_pos, all_agent_v, all_obstacle_pos])
        # agent_specific_global_state: Dict like {'adversary0':xx, 'agent':xx}.
        #               Shape:
        #               - good_agent: 2+2+2*num_obstacles+2*(num_good+num_adversaries-1)+2*(num_good-1)+(2+2)*(num_good+num_adversaries)+2*num_obstacles
        #               - adversary: 2+2+2*num_obstacles+2*(num_good+num_adversaries-1)+2*num_good+(2+2)*(num_good+num_adversaries)+2*num_obstacles
        #               2-dim vector. contains
        #               - agent_state info
        #               - global_state info
        if self._agent_specific_global_state:
            temp_global_state = {}
            for key in obs:
                temp_global_state[key] = np.concatenate([obs[key], ret['global_state']])
            ret['global_state'] = temp_global_state
        # agent_alone_state: Dict like {'adversary0':xx, 'agent':xx}
        #                    Shape: (2+2+2*num_obstacles,)
        #                    Stacked observation. Exclude other agents' positions from agent_state. Contains
        #                    - agent itself's state(velocity + position) +
        #                    - obstacles' positions (do not include other agents' positions)
        ret['agent_alone_state']={}
        for key in obs:
            ret['agent_alone_state'][key] = obs[key][0:4+2*self._num_obstacles]
        # agent_alone_padding_state: Dict like {'adversary0':xx, 'agent':xx}
        #                            Shape:
        #                            - good_agent: 2+2+2*num_obstacles+2*(num_good+num_adversaries-1)+2*(num_good-1)
        #                            - adversary: 2+2+2*num_obstacles+2*(num_good+num_adversaries-1)+2*num_good
        #                            Contains the same information as agent_alone_state;
        #                            But 0-padding other agents' positions.
        ret['agent_alone_padding_state'] = {}
        for key in obs:
            length = len(obs[key])
            new_value = np.zeros(length)
            new_value[0:4+2*self._num_obstacles] = obs[key][0:4+2*self._num_obstacles]
            ret['agent_alone_padding_state'][key] = new_value
        # action_mask: All actions are of use(either 1 for discrete or 5 for continuous). Thus all 1.
        ret['action_mask'] = np.ones((self._num_good+self._num_adversaries, *self._action_dim)).astype(np.float32)
        return ret

    def _process_action(self, action: 'torch.Tensor') -> Dict[str, np.ndarray]:  # noqa
        dict_action = {}
        for i, agent in enumerate(self._agents):
            agent_action = action[i]
            if agent_action.shape == (1, ):
                agent_action = agent_action.squeeze()  # 0-dim array
            dict_action[agent] = agent_action
        return dict_action

    def random_action(self) -> np.ndarray:
        random_action = self.action_space.sample()
        for k in random_action:
            if isinstance(random_action[k], np.ndarray):
                pass
            elif isinstance(random_action[k], int):
                random_action[k] = to_ndarray([random_action[k]], dtype=np.int64)
        return random_action

    def __repr__(self) -> str:
        return "DI-engine PettingZoo Env"

    @property
    def agents(self) -> List[str]:
        return self._agents

    @property
    def observation_space(self) -> gym.spaces.Space:
        return self._observation_space

    @property
    def action_space(self) -> gym.spaces.Space:
        return self._action_space

    @property
    def reward_space(self) -> gym.spaces.Space:
        return self._reward_space


class simple_tag_raw_env(SimpleEnv):

    def __init__(self, num_good=1, num_adversaries=3, num_obstacles=2, max_cycles=25, continuous_actions=False):
        scenario = Scenario()
        world = scenario.make_world(num_good,num_adversaries,num_obstacles)
        super().__init__(scenario, world, max_cycles, continuous_actions=continuous_actions)
        self.metadata['name'] = "simple_tag_v2"

    # def _execute_world_step(self):
    #     # set action for each agent
    #     for i, agent in enumerate(self.world.agents):
    #         action = self.current_actions[i]
    #         scenario_action = []
    #         if agent.movable:
    #             mdim = self.world.dim_p * 2 + 1
    #             if self.continuous_actions:
    #                 scenario_action.append(action[0:mdim])
    #                 action = action[mdim:]
    #             else:
    #                 scenario_action.append(action % mdim)
    #                 action //= mdim
    #         if not agent.silent:
    #             scenario_action.append(action)
    #         self._set_action(scenario_action, agent, self.action_spaces[agent.name])

    #     self.world.step()

    #     global_reward = 0.
    #     if self.local_ratio is not None:
    #         global_reward = float(self.scenario.global_reward(self.world))

    #     for agent in self.world.agents:
    #         agent_reward = float(self.scenario.reward(agent, self.world))
    #         if self.local_ratio is not None:
    #             # we changed reward calc way to keep same with mpe
    #             # reward = global_reward * (1 - self.local_ratio) + agent_reward * self.local_ratio
    #             reward = global_reward + agent_reward
    #         else:
    #             reward = agent_reward

    #         self.rewards[agent.name] = reward
