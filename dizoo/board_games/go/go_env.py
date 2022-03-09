import os

import numpy as np
import pygame
from gym import spaces

from pettingzoo import AECEnv
from pettingzoo.utils import wrappers
from pettingzoo.utils.agent_selector import agent_selector

from . import coords, go


def get_image(path):
    from os import path as os_path

    import pygame
    cwd = os_path.dirname(__file__)
    image = pygame.image.load(cwd + '/' + path)
    sfc = pygame.Surface(image.get_size(), flags=pygame.SRCALPHA)
    sfc.blit(image, (0, 0))
    return sfc


def env(**kwargs):
    env = raw_env(**kwargs)
    env = wrappers.TerminateIllegalWrapper(env, illegal_reward=-1)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env


class raw_env(AECEnv):

    metadata = {
        "render.modes": ["human", "rgb_array"],
        "name": "go_v5",
        "is_parallelizable": False,
        "video.frames_per_second": 2,
    }

    def __init__(self, board_size: int = 19, komi: float = 7.5):
        # board_size: a int, representing the board size (board has a board_size x board_size shape)
        # komi: a float, representing points given to the second player.
        super().__init__()

        self._overwrite_go_global_variables(board_size=board_size)
        self._komi = komi

        self.agents = ['black_0', 'white_0']
        self.possible_agents = self.agents[:]
        self.has_reset = False

        self.screen = None

        self.observation_spaces = self._convert_to_dict(
            [spaces.Dict({'observation': spaces.Box(low=0, high=1, shape=(self._N, self._N, 17), dtype=bool),
                          'action_mask': spaces.Box(low=0, high=1, shape=((self._N * self._N) + 1,), dtype=np.int8)})
             for _ in range(self.num_agents)])

        self.action_spaces = self._convert_to_dict([spaces.Discrete(self._N * self._N + 1) for _ in range(self.num_agents)])

        self._agent_selector = agent_selector(self.agents)

        self.board_history = np.zeros((self._N, self._N, 16), dtype=bool)

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def _overwrite_go_global_variables(self, board_size: int):
        self._N = board_size
        go.N = self._N
        go.ALL_COORDS = [(i, j) for i in range(self._N) for j in range(self._N)]
        go.EMPTY_BOARD = np.zeros([self._N, self._N], dtype=np.int8)
        go.NEIGHBORS = {(x, y): list(filter(self._check_bounds, [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)])) for x, y in go.ALL_COORDS}
        go.DIAGONALS = {(x, y): list(filter(self._check_bounds, [(x + 1, y + 1), (x + 1, y - 1), (x - 1, y + 1), (x - 1, y - 1)])) for x, y in go.ALL_COORDS}
        return

    def _check_bounds(self, c):
        return 0 <= c[0] < self._N and 0 <= c[1] < self._N

    def _encode_player_plane(self, agent):
        if agent == self.possible_agents[0]:
            return np.zeros([self._N, self._N], dtype=bool)
        else:
            return np.ones([self._N, self._N], dtype=bool)

    def _encode_board_planes(self, agent):
        agent_factor = go.BLACK if agent == self.possible_agents[0] else go.WHITE
        current_agent_plane_idx = np.where(self._go.board == agent_factor)
        opponent_agent_plane_idx = np.where(self._go.board == -agent_factor)
        current_agent_plane = np.zeros([self._N, self._N], dtype=bool)
        opponent_agent_plane = np.zeros([self._N, self._N], dtype=bool)
        current_agent_plane[current_agent_plane_idx] = 1
        opponent_agent_plane[opponent_agent_plane_idx] = 1
        return current_agent_plane, opponent_agent_plane

    def _int_to_name(self, ind):
        return self.possible_agents[ind]

    def _name_to_int(self, name):
        return self.possible_agents.index(name)

    def _convert_to_dict(self, list_of_list):
        return dict(zip(self.possible_agents, list_of_list))

    def _encode_legal_actions(self, actions):
        return np.where(actions == 1)[0]

    def _encode_rewards(self, result):
        return [1, -1] if result == 1 else [-1, 1]

    def observe(self, agent):
        current_agent_plane, opponent_agent_plane = self._encode_board_planes(agent)
        player_plane = self._encode_player_plane(agent)

        observation = np.dstack((self.board_history, player_plane))

        legal_moves = self.next_legal_moves if agent == self.agent_selection else []
        action_mask = np.zeros((self._N * self._N) + 1, 'int8')
        for i in legal_moves:
            action_mask[i] = 1

        return {'observation': observation, 'action_mask': action_mask}

    def step(self, action):
        if self.dones[self.agent_selection]:
            return self._was_done_step(action)
        self._go = self._go.play_move(coords.from_flat(action))
        self._last_obs = self.observe(self.agent_selection)
        current_agent_plane, opponent_agent_plane = self._encode_board_planes(self.agent_selection)
        self.board_history = np.dstack((current_agent_plane, opponent_agent_plane, self.board_history[:, :, :-2]))
        next_player = self._agent_selector.next()
        if self._go.is_game_over():
            self.dones = self._convert_to_dict([True for _ in range(self.num_agents)])
            self.rewards = self._convert_to_dict(self._encode_rewards(self._go.result()))
            self.next_legal_moves = [self._N * self._N]
        else:
            self.next_legal_moves = self._encode_legal_actions(self._go.all_legal_moves())
        self.agent_selection = next_player if next_player else self._agent_selector.next()
        self._accumulate_rewards()

    def reset(self):
        self.has_reset = True
        self._go = go.Position(board=None, komi=self._komi)

        self.agents = self.possible_agents[:]
        self._agent_selector.reinit(self.agents)
        self.agent_selection = self._agent_selector.reset()
        self._cumulative_rewards = self._convert_to_dict(np.array([0.0, 0.0]))
        self.rewards = self._convert_to_dict(np.array([0.0, 0.0]))
        self.dones = self._convert_to_dict([False for _ in range(self.num_agents)])
        self.infos = self._convert_to_dict([{} for _ in range(self.num_agents)])
        self.next_legal_moves = self._encode_legal_actions(self._go.all_legal_moves())
        self._last_obs = self.observe(self.agents[0])
        self.board_history = np.zeros((self._N, self._N, 16), dtype=bool)

    def render(self, mode='human'):
        screen_width = 1026
        screen_height = 1026

        if self.screen is None:
            if mode == "human":
                pygame.init()
                self.screen = pygame.display.set_mode((screen_width, screen_height))
            else:
                self.screen = pygame.Surface((screen_width, screen_height))
        if mode == "human":
            pygame.event.get()

        size = go.N

        # Load and scale all of the necessary images
        tile_size = (screen_width) / size

        black_stone = get_image(os.path.join('img', 'GoBlackPiece.png'))
        black_stone = pygame.transform.scale(black_stone, (int(tile_size * (5 / 6)), int(tile_size * (5 / 6))))

        white_stone = get_image(os.path.join('img', 'GoWhitePiece.png'))
        white_stone = pygame.transform.scale(white_stone, (int(tile_size * (5 / 6)), int(tile_size * (5 / 6))))

        tile_img = get_image(os.path.join('img', 'GO_Tile0.png'))
        tile_img = pygame.transform.scale(tile_img, ((int(tile_size * (7 / 6))), int(tile_size * (7 / 6))))

        # blit board tiles
        for i in range(1, size - 1):
            for j in range(1, size - 1):
                self.screen.blit(tile_img, ((i * (tile_size)), int(j) * (tile_size)))

        for i in range(1, 9):
            tile_img = get_image(os.path.join('img', 'GO_Tile' + str(i) + '.png'))
            tile_img = pygame.transform.scale(tile_img, ((int(tile_size * (7 / 6))), int(tile_size * (7 / 6))))
            for j in range(1, size - 1):
                if i == 1:
                    self.screen.blit(tile_img, (0, int(j) * (tile_size)))
                elif i == 2:
                    self.screen.blit(tile_img, ((int(j) * (tile_size)), 0))
                elif i == 3:
                    self.screen.blit(tile_img, ((size - 1) * (tile_size), int(j) * (tile_size)))
                elif i == 4:
                    self.screen.blit(tile_img, ((int(j) * (tile_size)), (size - 1) * (tile_size)))
            if i == 5:
                self.screen.blit(tile_img, (0, 0))
            elif i == 6:
                self.screen.blit(tile_img, ((size - 1) * (tile_size), 0))
            elif i == 7:
                self.screen.blit(tile_img, ((size - 1) * (tile_size), (size - 1) * (tile_size)))
            elif i == 8:
                self.screen.blit(tile_img, (0, (size - 1) * (tile_size)))

        offset = tile_size * (1 / 6)
        # Blit the necessary chips and their positions
        for i in range(0, size):
            for j in range(0, size):
                if self._go.board[i][j] == go.BLACK:
                    self.screen.blit(black_stone, ((i * (tile_size) + offset), int(j) * (tile_size) + offset))
                elif self._go.board[i][j] == go.WHITE:
                    self.screen.blit(white_stone, ((i * (tile_size) + offset), int(j) * (tile_size) + offset))

        if mode == "human":
            pygame.display.update()

        observation = np.array(pygame.surfarray.pixels3d(self.screen))

        return np.transpose(observation, axes=(1, 0, 2)) if mode == "rgb_array" else None

    def close(self):
        pass
