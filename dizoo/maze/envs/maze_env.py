from typing import List

import copy
import numpy as np
import gym
from gym import spaces
from gym.utils import seeding

from ding.envs import BaseEnvTimestep
from ding.utils import ENV_REGISTRY


@ENV_REGISTRY.register('maze')
class Maze(gym.Env):
    """
        Environment with random maze layouts. The ASCII representation of the mazes include the following objects:
      - `<SPACE>`: empty
      - `x`: wall
      - `S`: the start location (optional)
      - `T`: the target location.
      """
    KEY_EMPTY = 0
    KEY_WALL = 1
    KEY_TARGET = 2
    KEY_START = 3
    ASCII_MAP = {
        KEY_EMPTY: ' ',
        KEY_WALL: 'x',
        KEY_TARGET: 'T',
        KEY_START: 'S',
    }

    def __init__(
        self,
        cfg,
    ):
        self._size = cfg.size
        self._init_flag = False
        self._random_start = True
        self._seed = None
        self._step = 0

    def reset(self):
        self.active_init()
        obs = self._get_obs()
        self._step = 0
        return self.process_states(obs, self.get_maze_map())

    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        self._seed = seed
        self._dynamic_seed = dynamic_seed
        np.random.seed(self._seed)

    def active_init(
        self,
        tabular_obs=False,
        reward_fn=lambda x, y, tx, ty: 1 if (x == tx and y == ty) else 0,
        done_fn=lambda x, y, tx, ty: x == tx and y == ty
    ):
        self._maze = self.generate_maze(self.size, self._seed, 'tunnel')
        self._num_maze_keys = len(Maze.ASCII_MAP.keys())
        nav_map = self.maze_to_ascii(self._maze)
        self._map = nav_map
        self._tabular_obs = tabular_obs
        self._reward_fn = reward_fn
        self._done_fn = done_fn
        if self._reward_fn is None:
            self._reward_fn = lambda x, y, tx, ty: float(x == tx and y == ty)
        if self._done_fn is None:
            self._done_fn = lambda x, y, tx, ty: False

        self._max_x = len(self._map)
        if not self._max_x:
            raise ValueError('Invalid map.')
        self._max_y = len(self._map[0])
        if not all(len(m) == self._max_y for m in self._map):
            raise ValueError('Invalid map.')
        self._start_x, self._start_y = self._find_initial_point()
        self._target_x, self._target_y = self._find_target_point()
        self._x, self._y = self._start_x, self._start_y

        self._n_state = self._max_x * self._max_y
        self._n_action = 4

        if self._tabular_obs:
            self.observation_space = spaces.Discrete(self._n_state)
        else:
            self.observation_space = spaces.Box(low=0.0, high=np.inf, shape=(16, 16, 3))

        self.action_space = spaces.Discrete(self._n_action)
        self.reward_space = spaces.Box(low=0, high=1, shape=(1, ), dtype=np.float32)

    def random_start(self):
        init_x, init_y = self._x, self._y
        while True:  # Find empty grid cell.
            self._x = self.np_random.integers(self._max_x)
            self._y = self.np_random.integers(self._max_y)
            if self._map[self._x][self._y] != 'x':
                break
        ret = copy.deepcopy(self.process_states(self._get_obs(), self.get_maze_map()))
        self._x, self._y = init_x, init_y
        return ret

    def close(self) -> None:
        if self._init_flag:
            self._env.close()
        self._init_flag = False

    @property
    def num_maze_keys(self):
        return self._num_maze_keys

    @property
    def size(self):
        return self._size

    def process_states(self, observations, maze_maps):
        """Returns [B, W, W, 3] binary values. Channels are (wall; goal; obs)"""
        loc = np.eye(self._size * self._size, dtype=np.int64)[observations[0] * self._size + observations[1]]
        loc = np.reshape(loc, [self._size, self._size])
        maze_maps = maze_maps.astype(np.int64)

        states = np.concatenate([maze_maps, loc[Ellipsis, None]], axis=-1, dtype=np.int64)
        return states

    def get_maze_map(self, stacked=True):
        if not stacked:
            return self._maze.copy()
        wall = self._maze.copy()
        target_x, target_y = self.target_location
        assert wall[target_x][target_y] == Maze.KEY_TARGET
        wall[target_x][target_y] = 0
        target = np.zeros((self._size, self._size))
        target[target_x][target_y] = 1
        assert wall[self._start_x][self._start_y] == Maze.KEY_START
        wall[self._start_x][self._start_y] = 0
        return np.stack([wall, target], axis=-1)

    def generate_maze(self, size, seed, wall_type):
        rng, _ = seeding.np_random(seed)
        maze = np.full((size, size), fill_value=Maze.KEY_EMPTY, dtype=int)

        if wall_type == 'none':
            maze[[0, -1], :] = Maze.KEY_WALL
            maze[:, [0, -1]] = Maze.KEY_WALL
        elif wall_type == 'tunnel':
            self.sample_wall(maze, rng)
        elif wall_type.startswith('blocks:'):
            maze[[0, -1], :] = Maze.KEY_WALL
            maze[:, [0, -1]] = Maze.KEY_WALL
            self.sample_blocks(maze, rng, int(wall_type.split(':')[-1]))
        else:
            raise ValueError('Unknown wall type: %s' % wall_type)

        loc_target = self.sample_location(maze, rng)
        maze[loc_target] = Maze.KEY_TARGET

        loc_start = self.sample_location(maze, rng)
        maze[loc_start] = Maze.KEY_START
        self._start_x, self._start_y = loc_start

        return maze

    def sample_blocks(self, maze, rng, num_blocks):
        """Sample single-block 'wall' or 'obstacles'."""
        for _ in range(num_blocks):
            loc = self.sample_location(maze, rng)
            maze[loc] = Maze.KEY_WALL

    def sample_wall(
        self, maze, rng, shortcut_prob=0.1, inner_wall_thickness=1, outer_wall_thickness=1, corridor_thickness=2
    ):
        room = maze

        # step 1: fill everything as wall
        room[:] = Maze.KEY_WALL

        # step 2: prepare
        # we move two pixels at a time, because the walls are also occupying pixels
        delta = inner_wall_thickness + corridor_thickness
        dx = [delta, -delta, 0, 0]
        dy = [0, 0, delta, -delta]

        def get_loc_type(y, x):
            # remember there is a outside wall of 1 pixel surrounding the room
            if (y < outer_wall_thickness or y + corridor_thickness - 1 >= room.shape[0] - outer_wall_thickness):
                return 'invalid'
            if (x < outer_wall_thickness or x + corridor_thickness - 1 >= room.shape[1] - outer_wall_thickness):
                return 'invalid'
            # already visited
            if room[y, x] == Maze.KEY_EMPTY:
                return 'occupied'
            return 'valid'

        def connect_pixel(y, x, ny, nx):
            pixel = Maze.KEY_EMPTY
            if ny == y:
                room[y:y + corridor_thickness, min(x, nx):max(x, nx) + corridor_thickness] = pixel
            else:
                room[min(y, ny):max(y, ny) + corridor_thickness, x:x + corridor_thickness] = pixel

        def carve_passage_from(y, x):
            room[y, x] = Maze.KEY_EMPTY
            for direction in rng.permutation(len(dx)):
                ny = y + dy[direction]
                nx = x + dx[direction]

                loc_type = get_loc_type(ny, nx)
                if loc_type == 'invalid':
                    continue
                elif loc_type == 'valid':
                    connect_pixel(y, x, ny, nx)
                    # recursion
                    carve_passage_from(ny, nx)
                else:
                    # occupied
                    # we create shortcut with some probability, this is because
                    # we do not want to restrict to only one feasible path.
                    if rng.random() < shortcut_prob:
                        connect_pixel(y, x, ny, nx)

        carve_passage_from(outer_wall_thickness, outer_wall_thickness)

    def sample_location(self, maze, rng):
        for _ in range(1000):
            x, y = rng.integers(low=1, high=self._size, size=2)
            if maze[x, y] == Maze.KEY_EMPTY:
                return x, y
        raise ValueError('Cannot sample empty location, make maze bigger?')

    @staticmethod
    def key_to_ascii(key):
        if key in Maze.ASCII_MAP:
            return Maze.ASCII_MAP[key]
        assert (Maze.KEY_OBJ <= key < Maze.KEY_OBJ + Maze.MAX_OBJ_TYPES)
        return chr(ord('1') + key - Maze.KEY_OBJ)

    def maze_to_ascii(self, maze):
        return [[Maze.key_to_ascii(x) for x in row] for row in maze]

    def tabular_obs_action(self, status_obs, action, include_maze_layout=False):
        tabular_obs = self.get_tabular_obs(status_obs)
        multiplier = self._n_action
        if include_maze_layout:
            multiplier += self._num_maze_keys
        return multiplier * tabular_obs + action

    @staticmethod
    def create_collector_env_cfg(cfg: dict) -> List[dict]:
        collector_env_num = cfg.pop('collector_env_num')
        cfg = copy.deepcopy(cfg)
        cfg.is_train = True
        return [cfg for _ in range(collector_env_num)]

    @staticmethod
    def create_evaluator_env_cfg(cfg: dict) -> List[dict]:
        evaluator_env_num = cfg.pop('evaluator_env_num')
        cfg = copy.deepcopy(cfg)
        cfg.is_train = False
        return [cfg for _ in range(evaluator_env_num)]

    @property
    def nav_map(self):
        return self._map

    @property
    def n_state(self):
        return self._n_state

    @property
    def n_action(self):
        return self._n_action

    @property
    def target_location(self):
        return self._target_x, self._target_y

    @property
    def tabular_obs(self):
        return self._tabular_obs

    def _find_initial_point(self):
        for x in range(self._max_x):
            for y in range(self._max_y):
                if self._map[x][y] == 'S':
                    break
            if self._map[x][y] == 'S':
                break
        else:
            return None, None

        return x, y

    def _find_target_point(self):
        for x in range(self._max_x):
            for y in range(self._max_y):
                if self._map[x][y] == 'T':
                    break
            if self._map[x][y] == 'T':
                break
        else:
            raise ValueError('Target point not found in map.')

        return x, y

    def _get_obs(self):
        if self._tabular_obs:
            return self._x * self._max_y + self._y
        else:
            return np.array([self._x, self._y])

    def get_tabular_obs(self, status_obs):
        return self._max_y * status_obs[..., 0] + status_obs[..., 1]

    def get_xy(self, state):
        x = state / self._max_y
        y = state % self._max_y
        return x, y

    def step(self, action):
        last_x, last_y = self._x, self._y
        if action == 0:
            if self._x < self._max_x - 1:
                self._x += 1
        elif action == 1:
            if self._y < self._max_y - 1:
                self._y += 1
        elif action == 2:
            if self._x > 0:
                self._x -= 1
        elif action == 3:
            if self._y > 0:
                self._y -= 1

        if self._map[self._x][self._y] == 'x':
            self._x, self._y = last_x, last_y
        self._step += 1
        reward = self._reward_fn(self._x, self._y, self._target_x, self._target_y)
        done = self._done_fn(self._x, self._y, self._target_x, self._target_y)
        info = {}
        if self._step > 100:
            done = True
        if done:
            info['final_eval_reward'] = reward
            info['eval_episode_return'] = reward
        return BaseEnvTimestep(self.process_states(self._get_obs(), self.get_maze_map()), reward, done, info)


def get_value_map(env):
    """Returns [W, W, A] one-hot VI actions."""
    target_location = env.target_location
    nav_map = env.nav_map
    current_points = [target_location]
    chosen_actions = {target_location: 0}
    visited_points = {target_location: True}

    while current_points:
        next_points = []
        for point_x, point_y in current_points:
            for (action, (next_point_x, next_point_y)) in [(0, (point_x - 1, point_y)), (1, (point_x, point_y - 1)),
                                                           (2, (point_x + 1, point_y)), (3, (point_x, point_y + 1))]:

                if (next_point_x, next_point_y) in visited_points:
                    continue

                if not (0 <= next_point_x < len(nav_map) and 0 <= next_point_y < len(nav_map[next_point_x])):
                    continue

                if nav_map[next_point_x][next_point_y] == 'x':
                    continue

                next_points.append((next_point_x, next_point_y))
                visited_points[(next_point_x, next_point_y)] = True
                chosen_actions[(next_point_x, next_point_y)] = action
        current_points = next_points

    value_map = np.zeros([env.size, env.size, env.n_action])
    for (x, y), action in chosen_actions.items():
        value_map[x][y][action] = 1
    return value_map
