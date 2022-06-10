import gym
import numpy as np
import sys

from dizoo.board_games.base_game_env import BaseGameEnv
from ding.envs import BaseEnv, BaseEnvTimestep
from ding.envs.common.env_element import EnvElement, EnvElementInfo
from ding.utils import ENV_REGISTRY


@ENV_REGISTRY.register('gomoku')
class GomokuEnv(BaseGameEnv):
    def __init__(self, cfg={}):
        self.cfg = cfg
        self.board_size = self.cfg.get('board_size', 15)
        self.players = [1, 2]
        # self.board_markers = [
        #     chr(x) for x in range(ord("A"), ord("A") + self.board_size)
        # ]
        self.board_markers = [
            str(i + 1) for i in range(self.board_size)
        ]
        self.num_actions = self.board_size * self.board_size

    @property
    def current_player(self):
        return self._current_player

    @property
    def current_opponent_player(self):
        return self.players[0] if self.current_player == self.players[1] else self.players[1]

    @property
    def legal_actions(self):
        legal_actions = []
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.board[i][j] == 0:
                    legal_actions.append(self.coord_to_action(i, j))
        return legal_actions

    def reset(self, start_player=0):
        self._observation_space = gym.spaces.Box(
            low=0, high=2, shape=(self.board_size, self.board_size, 3), dtype=np.int32
        )
        self._action_space = gym.spaces.Discrete(self.board_size ** 2)
        self._reward_space = gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)

        self._current_player = self.players[start_player]
        self.board = np.zeros((self.board_size, self.board_size), dtype="int32")
        self._final_eval_reward = 0.
        # return self.current_state()
        action_mask = np.zeros(self.num_actions, 'int8')
        action_mask[self.legal_actions] = 1
        obs = {'observation': self.current_state(), 'action_mask': action_mask}
        return BaseEnvTimestep(obs, None, None, None)

    def do_action(self, action):
        row, col = self.action_to_coord(action)
        self.board[row, col] = self.current_player
        self._current_player = self.current_opponent_player

    def step(self, action):
        curr_player = self.current_player
        next_player = self.current_opponent_player
        if action in self.legal_actions:
            self.do_action(action)
        else:
            print("Error: input illegal action, we randomly choice a action from self.legal_actions!")
            action = np.random.choice(self.legal_actions)
            self.do_action(action)
            # sys.exit(-1)

        done, winner = self.game_end()
        reward = int((winner == curr_player))
        info = {'next player to play': next_player}

        if done:
            self._final_eval_reward = reward
            info['final_eval_reward'] = self._final_eval_reward
            print('gomoku one episode done: ', info)

        action_mask = np.zeros(self.num_actions, 'int8')
        action_mask[self.legal_actions] = 1
        obs = {'observation': self.current_state(), 'action_mask': action_mask}
        return BaseEnvTimestep(obs, reward, done, info)

    def current_state(self):
        board_curr_player = np.where(self.board == self.current_player, 1, 0)
        board_opponent_player = np.where(self.board == self.current_opponent_player, 1, 0)
        board_to_play = np.full((self.board_size, self.board_size), self.current_player)
        return np.array([board_curr_player, board_opponent_player, board_to_play], dtype=np.float32)

    def coord_to_action(self, i, j):
        ''' convert coordinate i, j to action a in [0, board_size**2)
        '''
        a = i * self.board_size + j  # action index
        return a

    def action_to_coord(self, a):
        coord = (a // self.board_size, a % self.board_size)
        return coord

    def have_winner(self):
        has_legal_actions = False
        directions = ((1, -1), (1, 0), (1, 1), (0, 1))
        for i in range(self.board_size):
            for j in range(self.board_size):
                # if no stone is on the position, don't need to consider this position
                if self.board[i][j] == 0:
                    has_legal_actions = True
                    continue
                # value-value at a coord, i-row, j-col
                player = self.board[i][j]
                # check if there exist 5 in a line
                for d in directions:
                    x, y = i, j
                    count = 0
                    for _ in range(5):
                        if (x not in range(self.board_size)) or (
                                y not in range(self.board_size)
                        ):
                            break
                        if self.board[x][y] != player:
                            break
                        x += d[0]
                        y += d[1]
                        count += 1
                        # if 5 in a line, store positions of all stones, return value
                        if count == 5:
                            return True, player
        return not has_legal_actions, -1

    def game_end(self):
        end, winner = self.have_winner()
        return end, winner

    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        self._seed = seed
        self._dynamic_seed = dynamic_seed
        np.random.seed(self._seed)

    def random_action(self):
        action_list = self.legal_actions
        return np.random.choice(action_list)

    def render(self):
        marker = "   "
        for i in range(self.board_size):
            if i <= 8:
                marker = marker + self.board_markers[i] + "  "
            else:
                marker = marker + self.board_markers[i] + " "
        print(marker)
        for row in range(self.board_size):
            # print(chr(ord("A") + row), end=" ")
            if row <= 8:
                print(str(1 + row) + ' ', end=" ")
            else:
                print(str(1 + row), end=" ")
            for col in range(self.board_size):
                ch = self.board[row][col]
                if ch == 0:
                    print(".", end="  ")
                elif ch == 1:
                    print("X", end="  ")
                elif ch == 2:
                    print("O", end="  ")
            print()

    def human_to_action(self):
        """
        For multiplayer games, ask the user for a legal action
        and return the corresponding action number.
        Returns:
            An integer from the action space.
        """
        while True:
            try:
                row = int(
                    input(
                        f"Enter the row (1, 2, ...,{self.board_size}, from up to bottom) to play for the player {self.current_player}: "
                    )
                )
                col = int(
                    input(
                        f"Enter the column (1, 2, ...,{self.board_size}, from left to right) to play for the player {self.current_player}: "
                    )
                )
                choice = self.coord_to_action(row - 1, col - 1)
                if (
                        choice in self.legal_actions
                        and 1 <= row
                        and 1 <= col
                        and row <= self.board_size
                        and col <= self.board_size
                ):
                    break
                else:
                    print("Wrong input, try again")
            except KeyboardInterrupt:
                print("exit")
                sys.exit(0)
            except Exception as e:
                print("Wrong input, try again")
        return choice

    def action_to_string(self, action_number):
        """
        Convert an action number to a string representing the action.

        Args:
            action_number: an integer from the action space.
        Returns:
            String representing the action.
        """
        row = action_number // self.board_size + 1
        col = action_number % self.board_size + 1
        return f"Play row {row}, column {col}"

    def get_equi_data(self, play_data):
        """augment the data set by rotation and flipping
        play_data: [(state, mcts_prob, winner_z), ..., ...]
        """
        extend_data = []
        for data in play_data:
            state = data['state']
            mcts_prob = data['mcts_prob']
            winner = data['winner']
            for i in [1, 2, 3, 4]:
                # rotate counterclockwise
                equi_state = np.array([np.rot90(s, i) for s in state])
                equi_mcts_prob = np.rot90(np.flipud(
                    mcts_prob.reshape(self.board_size, self.board_size)), i)
                extend_data.append({'state': equi_state,
                                    'mcts_prob': np.flipud(equi_mcts_prob).flatten(),
                                    'winner': winner})
                # flip horizontally
                equi_state = np.array([np.fliplr(s) for s in equi_state])
                equi_mcts_prob = np.fliplr(equi_mcts_prob)
                extend_data.append({'state': equi_state,
                                    'mcts_prob': np.flipud(equi_mcts_prob).flatten(),
                                    'winner': winner})
        return extend_data

    @property
    def observation_space(self) -> gym.spaces.Space:
        return self._observation_space

    @property
    def action_space(self) -> gym.spaces.Space:
        return self._action_space

    @property
    def reward_space(self) -> gym.spaces.Space:
        return self._reward_space

    def close(self) -> None:
        pass

    def __repr__(self) -> str:
        return 'gomoku'


if __name__ == '__main__':
    env = GomokuEnv()
    obs = env.reset()
    print('init board state: ')
    env.render()
    done = False
    while True:
        action = env.random_action()
        # action = env.human_to_action()
        print('player 1: ' + env.action_to_string(action))
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            if reward > 0:
                print('player 1 (human player) win')
            else:
                print('draw')
            break

        action = env.random_action()
        print('player 2 (computer player): ' + env.action_to_string(action))
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            if reward > 0:
                print('player 2 (computer player) win')
            else:
                print('draw')
            break
