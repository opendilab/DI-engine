"""
Adapt Chess to BaseGameEnv interface from pettingzoo: https://github.com/Farama-Foundation/PettingZoo
"""

import chess
from gym import spaces
from pettingzoo.utils.agent_selector import agent_selector
from pettingzoo.classic.chess import chess_utils
import numpy as np
import sys
from dizoo.board_games.base_game_env import BaseGameEnv
from ding.envs import BaseEnvTimestep
from ding.utils import ENV_REGISTRY


@ENV_REGISTRY.register('Chess')
class ChessEnv(BaseGameEnv):

    def __init__(self, cfg=None):
        self.cfg = cfg
        self.current_player_index = 0
        self.next_player_index = 1

        self.board = chess.Board()

        self.agents = [f"player_{i+1}" for i in range(2)]
        self.possible_agents = self.agents[:]

        self._agent_selector = agent_selector(self.agents)

        self._action_spaces = {name: spaces.Discrete(8 * 8 * 73) for name in self.agents}
        self._observation_spaces = {
            name: spaces.Dict(
                {
                    'observation': spaces.Box(low=0, high=1, shape=(8, 8, 111), dtype=bool),
                    'action_mask': spaces.Box(low=0, high=1, shape=(4672, ), dtype=np.int8)
                }
            )
            for name in self.agents
        }

        self.rewards = None
        self.dones = None
        self.infos = {name: {} for name in self.agents}

        self.agent_selection = None

        self.board_history = np.zeros((8, 8, 104), dtype=bool)

    @property
    def current_player(self):
        return self.current_player_index

    def to_play(self):
        return self.next_player_index

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
        self.current_player_index = 0

        for agent, reward in self.rewards.items():
            self._cumulative_rewards[agent] += reward

        agent = self.agent_selection
        current_index = self.agents.index(agent)
        self.current_player_index = current_index
        obs = self.observe(agent)
        return obs

    def observe(self, agent):
        observation = chess_utils.get_observation(self.board, self.possible_agents.index(agent))
        observation = np.dstack((observation[:, :, :7], self.board_history))
        action_mask = self.legal_actions

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

        current_agent = self.agent_selection
        current_index = self.agents.index(current_agent)
        self.current_player_index = current_index

        next_board = chess_utils.get_observation(self.board, current_agent)
        self.board_history = np.dstack((next_board[:, :, 7:], self.board_history[:, :, :-13]))
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

        self.agent_selection = self._agent_selector.next()
        agent = self.agent_selection
        self.next_player_index = self.agents.index(agent)

        observation = self.observe(agent)

        return BaseEnvTimestep(observation, self._cumulative_rewards[agent], self.dones[agent], self.infos[agent])

    @property
    def legal_actions(self):
        action_mask = np.zeros(4672, 'uint8')
        action_mask[chess_utils.legal_moves(self.board)] = 1
        return action_mask  # 4672 dim {0,1}

    def legal_moves(self):
        legal_moves = chess_utils.legal_moves(self.board)
        return legal_moves

    def random_action(self):
        action_list = self.legal_moves()
        return np.random.choice(action_list)

    def expert_action(self):
        # TODO
        pass

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
                choice = int(input(f"Enter the index of next move for the player {self.to_play()}: "))
                if choice in self.legal_moves():
                    break
            except KeyboardInterrupt:
                sys.exit(0)
            except Exception as e:
                print("Wrong input, try again")
        return choice

    def render(self, mode='human'):
        print(self.board)

    @property
    def observation_space(self):
        return self._observation_spaces

    @property
    def action_space(self):
        return self._action_spaces

    @property
    def reward_space(self):
        return self._reward_space

    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        self._seed = seed
        self._dynamic_seed = dynamic_seed
        np.random.seed(self._seed)

    def close(self) -> None:
        pass

    def __repr__(self) -> str:
        return "DI-engine Chess Env"
