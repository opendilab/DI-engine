import chess
from gym import spaces
from pettingzoo.utils.agent_selector import agent_selector
from pettingzoo.classic.chess import chess_utils
import numpy as np
import sys
from dizoo.board_games.base_game_env import BaseGameEnv
from ding.envs import BaseEnv, BaseEnvInfo, BaseEnvTimestep
from ding.envs.common.env_element import EnvElement, EnvElementInfo
from ding.utils import ENV_REGISTRY

"""
Adapte Chess to DI-engine from pettingzoo: https://github.com/Farama-Foundation/PettingZoo
"""
@ENV_REGISTRY.register('ChessDI')
class ChessDIEnv(BaseGameEnv):
    def __init__(self, cfg=None):
        self.cfg = cfg
        self.current_player_index = 0

        self.board = chess.Board()

        self.agents = [f"player_{i}" for i in range(2)]
        self.possible_agents = self.agents[:]

        self._agent_selector = agent_selector(self.agents)

        self.action_spaces = {name: spaces.Discrete(8 * 8 * 73) for name in self.agents}
        self.observation_spaces = {name: spaces.Dict({
            'observation': spaces.Box(low=0, high=1, shape=(8, 8, 111), dtype=bool),
            'action_mask': spaces.Box(low=0, high=1, shape=(4672,), dtype=np.int8)
        }) for name in self.agents}

        self.rewards = None
        self.dones = None
        self.infos = {name: {} for name in self.agents}

        self.agent_selection = None

        self.board_history = np.zeros((8, 8, 104), dtype=bool)

    @property
    def current_player(self):
        return self.current_player_index

    def to_play(self):
        return self.current_player_index

    def reset(self):
        self.has_reset = True
        self.agents = self.possible_agents[:]
        self.board = chess.Board()

        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()

        self.rewards = {name: 0 for name in self.agents}
        self._cumulative_rewards = {name: 0 for name in self.agents}
        self.dones = {name: False for name in self.agents}
        self.infos = {name: {} for name in self.agents}

        self.board_history = np.zeros((8, 8, 104), dtype=bool)

        self.current_player_index = 0  # next player is 0

        for agent, reward in self.rewards.items():
            self._cumulative_rewards[agent] += reward

        agent = self.agent_selection
        current_index = self.agents.index(agent)
        self.current_player_index = current_index
        observation = self.observe(agent)
        return BaseEnvTimestep(observation, self._cumulative_rewards[agent], self.dones[agent], self.infos[agent])

    def observe(self, agent):
        observation = chess_utils.get_observation(self.board, self.possible_agents.index(agent))
        observation = np.dstack((observation[:, :, :7], self.board_history))
        legal_moves = chess_utils.legal_moves(self.board) if agent == self.agent_selection else []

        action_mask = np.zeros(4672, 'int8')
        for i in legal_moves:
            action_mask[i] = 1

        return {'observation': observation, 'action_mask': action_mask}

    def set_game_result(self, result_val):
        for i, name in enumerate(self.agents):
            self.dones[name] = True
            result_coef = 1 if i == 0 else -1
            self.rewards[name] = result_val * result_coef
            self.infos[name] = {'legal_moves': []}

    def step(self, action):

        if self.dones[self.agent_selection]:
            return self._was_done_step(action)
        current_agent = self.agent_selection  # 'player_0'
        current_index = self.agents.index(current_agent)  # 0
        self.current_player_index = current_index
        next_board = chess_utils.get_observation(self.board, current_agent)
        self.board_history = np.dstack((next_board[:, :, 7:], self.board_history[:, :, :-13]))
        self.agent_selection = self._agent_selector.next()

        chosen_move = chess_utils.action_to_move(self.board, action, current_index)
        assert chosen_move in self.board.legal_moves
        self.board.push(chosen_move)  # NOTE

        next_legal_moves = chess_utils.legal_moves(self.board)

        is_stale_or_checkmate = not any(next_legal_moves)

        # claim draw is set to be true to align with normal tournament rules
        is_repetition = self.board.is_repetition(3)
        is_50_move_rule = self.board.can_claim_fifty_moves()
        is_claimable_draw = is_repetition or is_50_move_rule
        game_over = is_claimable_draw or is_stale_or_checkmate

        if game_over:
            result = self.board.result(claim_draw=True)
            result_val = chess_utils.result_to_int(result)
            self.set_game_result(result_val)

        # self._accumulate_rewards()
        for agent, reward in self.rewards.items():
            self._cumulative_rewards[agent] += reward

        agent = self.agent_selection
        current_index = self.agents.index(agent)
        self.current_player_index = current_index

        observation = self.observe(agent)
        return BaseEnvTimestep(observation, self._cumulative_rewards[agent], self.dones[agent], self.infos[agent])

    def legal_actions(self):
        action_mask = np.zeros(4672, 'int8')
        for i in chess_utils.legal_moves(self.board):
            action_mask[i] = 1
        return action_mask  # 4672 dim {0,1}

    def legal_moves(self):
        legal_moves = chess_utils.legal_moves(self.board)

        return legal_moves

    def observation_space(self):
        return self.observation_spaces

    def action_space(self):
        return self.action_spaces

    def random_action(self):
        action_list = self.legal_moves()
        return np.random.choice(action_list)

    def render(self, mode='human'):
        print(self.board)

    def human_to_action(self):
        """
        For multiplayer games, ask the user for a legal action
        and return the corresponding action number.
        Returns:
            An integer from the action space.
        """
        while True:
            try:
                print(f"Current available actions for the player {self.to_play()} are:{self.legal_moves()}")
                choice = int(
                    input(
                        f"Enter the index of next move for the player {self.to_play()}: "
                    )
                )
                if choice in self.legal_moves():
                    break
            except KeyboardInterrupt:
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
        return str(action_number)

    def close(self) -> None:
        pass

    def game_end(self):
        """Check whether the game is ended or not"""
        pass

    def seed(self, seed: int) -> None:
        pass

    def do_action(self, action):
        pass

    def game_end(self):
        """Check whether the game is ended or not"""
        pass

    def __repr__(self) -> str:
        return 'chess'


if __name__ == '__main__':
    # from dizoo.board_games.chess.chess_env_di import ChessDIEnv
    env = ChessDIEnv()
    obs, reward, done, info = env.reset()
    env.render()
    while True:
        action = env.human_to_action()
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            env.render()
            if reward > 0:
                print('human player win')
            else:
                print('draw')
            break

        action = env.random_action()
        # print('computer player ' + env.action_to_string(action))
        print('computer player (player1) take action: ' + f'{action}')

        obs, reward, done, info = env.step(action)
        print(f'After the computer player (player1) took action: {action}, the current board state is:')
        env.render()
        if done:
            if reward > 0:
                print('computer player win')
            else:
                print('draw')
            break
