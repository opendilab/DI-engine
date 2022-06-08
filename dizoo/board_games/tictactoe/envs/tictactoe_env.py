import sys
import gym
import numpy as np
from ding.envs.env.base_env import BaseEnvTimestep
from ding.utils.registry_factory import ENV_REGISTRY
from dizoo.board_games.base_game_env import BaseGameEnv


@ENV_REGISTRY.register('tictactoe')
class TicTacToeEnv(BaseGameEnv):
    def __init__(self, cfg=None):
        self.board_size = 3
        self.players = [1, 2]
        self.num_actions = 9

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
        # self._observation_space = gym.spaces.Box(
        #     low=-1, high=1, shape=(self.board_size, self.board_size, 3), dtype=np.float32
        # )
        self._action_space = gym.spaces.Discrete(self.board_size ** 2)
        self._reward_space = gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

        self._current_player = self.players[start_player]
        self.board = np.zeros((self.board_size, self.board_size), dtype="int32")
        self._final_eval_reward = 0.
        # return self.current_state()

        action_mask = np.zeros(self.num_actions, 'int8')
        action_mask[self.legal_actions] = 1
        obs={'observation': self.current_state(), 'action_mask': action_mask}
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
            print("Error: illegal action")
            action = np.random.choice(self.legal_actions)
            self.do_action(action)
            # sys.exit(-1)

        done, winner = self.game_end()
        reward = int((winner == curr_player))
        info = {'next_player': next_player}

        if done:
            self._final_eval_reward = reward
            info['final_eval_reward'] = self._final_eval_reward
            print('tictactoe episode done: ', info)
        # return BaseEnvTimestep(self.current_state(), reward, done, info)
        action_mask = np.zeros(self.num_actions, 'int8')
        action_mask[self.legal_actions] = 1
        obs={'observation': self.current_state(), 'action_mask': action_mask}
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
        # Horizontal and vertical checks
        for i in range(self.board_size):
            if len(set(self.board[i, :])) == 1 and (self.board[i, 0] != 0):
                return True, self.board[i, 0]
            if len(set(self.board[:, i])) == 1 and (self.board[0, i] != 0):
                return True, self.board[0, i]

        # Diagonal checks
        if self.board[0, 0] == self.board[1, 1] == self.board[2, 2] != 0:
            return True, self.board[0, 0]
        if self.board[2, 0] == self.board[1, 1] == self.board[0, 2] != 0:
            return True, self.board[2, 0]

        return False, -1

    def game_end(self):
        """Check whether the game is ended or not"""
        win, winner = self.have_winner()
        if win:
            return True, winner
        elif len(self.legal_actions) == 0:
            return True, -1
        else:
            return False, -1

    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        self._seed = seed
        self._dynamic_seed = dynamic_seed
        np.random.seed(self._seed)

    def expert_action(self):
        board = self.board
        action = np.random.choice(self.legal_actions)
        # Horizontal and vertical checks
        for i in range(3):
            if abs(sum(board[i, :])) == 2:
                ind = np.where(board[i, :] == 0)[0][0]
                action = np.ravel_multi_index(
                    (np.array([i]), np.array([ind])), (3, 3)
                )[0]
                if self.current_player * sum(board[i, :]) > 0:
                    return action

            if abs(sum(board[:, i])) == 2:
                ind = np.where(board[:, i] == 0)[0][0]
                action = np.ravel_multi_index(
                    (np.array([ind]), np.array([i])), (3, 3)
                )[0]
                if self.current_player * sum(board[:, i]) > 0:
                    return action

        # Diagonal checks
        diag = board.diagonal()
        anti_diag = np.fliplr(board).diagonal()
        if abs(sum(diag)) == 2:
            ind = np.where(diag == 0)[0][0]
            action = np.ravel_multi_index(
                (np.array([ind]), np.array([ind])), (3, 3)
            )[0]
            if self.current_player * sum(diag) > 0:
                return action

        if abs(sum(anti_diag)) == 2:
            ind = np.where(anti_diag == 0)[0][0]
            action = np.ravel_multi_index(
                (np.array([ind]), np.array([2 - ind])), (3, 3)
            )[0]
            if self.current_player * sum(anti_diag) > 0:
                return action

        return action

    def render(self):
        print(self.board[::-1])

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
                        f"Enter the row (1, 2 or 3, from bottom to up) to play for the player {self.current_player}: "
                    )
                )
                col = int(
                    input(
                        f"Enter the column (1, 2 or 3, from left to right) to play for the player {self.current_player}: "
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
        return 'TicTacToe'


if __name__ == '__main__':
    env = TicTacToeEnv()
    env.reset()
    done = False
    while True:
        env.render()
        action = env.human_to_action()
        obs, reward, done, info = env.step(action)
        if done:
            env.render()
            if reward > 0:
                print('human player win')
            else:
                print('draw')
            break
        env.render()
        action = env.expert_action()
        print('computer player ' + env.action_to_string(action))
        obs, reward, done, info = env.step(action)
        if done:
            if reward > 0:
                print('computer player win')
            else:
                print('draw')
            break
