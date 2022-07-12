"""
Adapt TicTacToe to BaseGameEnv interface from https://github.com/werner-duvaud/muzero-general
"""
from ditk import logging
import sys
from typing import Any, List, Union, Sequence
import gym
import copy
import numpy as np

from ding.envs.env.base_env import BaseEnvTimestep
from ding.utils.registry_factory import ENV_REGISTRY
from dizoo.board_games.base_game_env import BaseGameEnv


@ENV_REGISTRY.register('tictactoe')
class TicTacToeEnv(BaseGameEnv):
    def __init__(self, cfg=None):
        self.battle_mode = cfg.battle_mode
        self.board_size = 3
        self.players = [1, 2]
        self.total_num_actions = 9

    @property
    def current_player(self):
        return self._current_player

    @property
    def to_play(self):
        return self.players[0] if self.current_player == self.players[1] else self.players[1]

    @property
    def current_player_to_compute_expert_action(self):
        """
        Overview: to compute expert action easily.
        """
        return -1 if self.current_player == 1 else 1

    def reset(self, start_player=0):
        self._observation_space = gym.spaces.Box(
            low=0, high=2, shape=(self.board_size, self.board_size, 3), dtype=np.uint8
        )
        self._action_space = gym.spaces.Discrete(self.board_size ** 2)
        self._reward_space = gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)

        self._current_player = self.players[start_player]
        self.board = np.zeros((self.board_size, self.board_size), dtype="int32")
        action_mask = np.zeros(self.total_num_actions, 'int8')
        action_mask[self.legal_actions] = 1
        if self.battle_mode == 'two_player_mode':
            obs = {'observation': self.current_state(), 'action_mask': action_mask, 'to_play': self.to_play}
        else:
            obs = {'observation': self.current_state(), 'action_mask': action_mask, 'to_play': None}
        return obs

    def step(self, action):
        if self.battle_mode == 'two_player_mode':
            timestep = self._player_step(action)
            return timestep
        elif self.battle_mode == 'one_player_mode':
            # player 1 battle with expert player 2

            # player 1's turn
            timestep_player1 = self._player_step(action)
            # self.env.render()
            if timestep_player1.done:
                timestep_player1.obs['to_play'] = None
                return timestep_player1

            # player 2's turn
            expert_action = self.expert_action()
            # print('player 2 (computer player): ' + self.action_to_string(expert_action))
            timestep_player2 = self._player_step(expert_action)
            # the final_eval_reward is calculated from Player 1's perspective
            timestep_player2.info['final_eval_reward'] = - timestep_player2.reward

            timestep = timestep_player2
            timestep.obs['to_play'] = None
            return timestep

    def _player_step(self, action):
        if action in self.legal_actions:
            row, col = self.action_to_coord(action)
            self.board[row, col] = self.current_player
        else:
            logging.warning(
                f"You input illegal action: {action}, the legal_actions are {self.legal_actions}. "
                f"Now we randomly choice a action from self.legal_actions."
            )
            action = np.random.choice(self.legal_actions)
            row, col = self.action_to_coord(action)
            self.board[row, col] = self.current_player

        # Check whether the game is ended or not and give the winner
        have_winner, winner = self.have_winner()
        if have_winner:
            done, winner = True, winner
        elif len(self.legal_actions) == 0:
            # the agent don't have legal_actions to move, so episode is done
            # winner=-1 indicates draw
            done, winner = True, -1
        else:
            # episode is not done
            done, winner = False, -1

        reward = np.array(float(winner == self.current_player)).astype(np.float32)
        info = {'next player to play': self.to_play}
        """
        NOTE: here exchange the player
        """
        self.current_player = self.to_play

        if done:
            info['final_eval_reward'] = reward
            # print('tictactoe one episode done: ', info)

        action_mask = np.zeros(self.total_num_actions, 'int8')
        action_mask[self.legal_actions] = 1
        obs = {'observation': self.current_state(), 'action_mask': action_mask, 'to_play': self.current_player}
        return BaseEnvTimestep(obs, reward, done, info)

    def current_state(self):
        board_curr_player = np.where(self.board == self.current_player, 1, 0)
        board_opponent_player = np.where(self.board == self.to_play, 1, 0)
        board_to_play = np.full((self.board_size, self.board_size), self.current_player)
        return np.array([board_curr_player, board_opponent_player, board_to_play], dtype=np.float32)

    def coord_to_action(self, i, j):
        """
        Overview:
            convert coordinate i, j to action index a in [0, board_size**2)
        """
        return i * self.board_size + j

    def action_to_coord(self, a):
        """
        Overview:
            convert action index a in [0, board_size**2) to coordinate (i, j)
        """
        return a // self.board_size, a % self.board_size

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

    @property
    def legal_actions(self):
        legal_actions = []
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.board[i][j] == 0:
                    legal_actions.append(self.coord_to_action(i, j))
        return legal_actions

    def random_action(self):
        action_list = self.legal_actions
        return np.random.choice(action_list)

    def expert_action(self):
        """
        Overview:
            Hard coded expert agent for tictactoe env
        Returns:
            - action (:obj:`int`): the expert action to take in the current game state.
        """
        # To easily calculate expert action, we convert the chessboard notation:
        # from player 1:  1, player 2: 2
        # to   player 1: -1, player 2: 1
        # TODO: more elegant implementation
        board = copy.deepcopy(self.board)
        for i in range(board.shape[0]):
            for j in range(board.shape[1]):
                if board[i][j] == 1:
                    board[i][j] = -1
                elif board[i][j] == 2:
                    board[i][j] = 1

        action = np.random.choice(self.legal_actions)
        # Horizontal and vertical checks
        for i in range(3):
            if abs(sum(board[i, :])) == 2:
                ind = np.where(board[i, :] == 0)[0][0]
                action = np.ravel_multi_index(
                    (np.array([i]), np.array([ind])), (3, 3)
                )[0]
                if self.current_player_to_compute_expert_action * sum(board[i, :]) > 0:
                    return action

            if abs(sum(board[:, i])) == 2:
                ind = np.where(board[:, i] == 0)[0][0]
                action = np.ravel_multi_index(
                    (np.array([ind]), np.array([i])), (3, 3)
                )[0]
                if self.current_player_to_compute_expert_action * sum(board[:, i]) > 0:
                    return action

        # Diagonal checks
        diag = board.diagonal()
        anti_diag = np.fliplr(board).diagonal()
        if abs(sum(diag)) == 2:
            ind = np.where(diag == 0)[0][0]
            action = np.ravel_multi_index(
                (np.array([ind]), np.array([ind])), (3, 3)
            )[0]
            if self.current_player_to_compute_expert_action * sum(diag) > 0:
                return action

        if abs(sum(anti_diag)) == 2:
            ind = np.where(anti_diag == 0)[0][0]
            action = np.ravel_multi_index(
                (np.array([ind]), np.array([2 - ind])), (3, 3)
            )[0]
            if self.current_player_to_compute_expert_action * sum(anti_diag) > 0:
                return action

        return action

    def human_to_action(self):
        """
        Overview:
            For multiplayer games, ask the user for a legal action
            and return the corresponding action number.
        Returns:
            An integer from the action space.
        """
        while True:
            try:
                row = int(
                    input(
                        f"Enter the row (1, 2, or 3, from up to bottom) to play for the player {self.current_player}: "

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

    def render(self, mode="human"):
        print(self.board)

    def action_to_string(self, action_number):
        """
        Overview:
            Convert an action number to a string representing the action.
        Arguments:
            - action_number: an integer from the action space.
        Returns:
            - String representing the action.
        """
        row = action_number // self.board_size + 1
        col = action_number % self.board_size + 1
        return f"Play row {row}, column {col}"

    @property
    def observation_space(self) -> gym.spaces.Space:
        return self._observation_space

    @property
    def action_space(self) -> gym.spaces.Space:
        return self._action_space

    @property
    def reward_space(self) -> gym.spaces.Space:
        return self._reward_space

    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        self._seed = seed
        self._dynamic_seed = dynamic_seed
        np.random.seed(self._seed)

    def close(self) -> None:
        pass

    @current_player.setter
    def current_player(self, value):
        self._current_player = value

    @staticmethod
    def create_collector_env_cfg(cfg: dict) -> List[dict]:
        collector_env_num = cfg.pop('collector_env_num')
        cfg = copy.deepcopy(cfg)
        return [cfg for _ in range(collector_env_num)]

    @staticmethod
    def create_evaluator_env_cfg(cfg: dict) -> List[dict]:
        evaluator_env_num = cfg.pop('evaluator_env_num')
        cfg = copy.deepcopy(cfg)
        cfg.battle_mode = 'one_player_mode'
        return [cfg for _ in range(evaluator_env_num)]

    def __repr__(self) -> str:
        return "DI-engine TicTacToe Env"
