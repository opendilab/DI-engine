import numpy as np
import sys

sys.path.extend(['/Users/puyuan/code/DI-engine', '/Users/puyuan/code/MuZero'])

one_episode_replay_buffer_tictocetoe_2_player_mode = np.load(
    '/Users/puyuan/code/DI-engine/dizoo/board_games/tictactoe/config/one_episode_replay_buffer_tictactoe_2-player-mode.npy', allow_pickle=True
)

"""
Note: please refer to ding/rl_utils/mcts/game.py for details.

game_history element shape:
    e.g. game_history_length=9, stack=4, num_unroll_steps=3, td_steps=2

    obs: game_history_length + stack + num_unroll_steps, 9+4 +3
    action: game_history_length -> 9
    reward: game_history_length + num_unroll_steps + td_steps -1  9 + 3 + 2-1
    root_values:  game_history_length + num_unroll_steps + td_steps -> 9 + 3 + 2
    child_visits： game_history_length + num_unroll_steps -> 9 +3
    to_play: game_history_length -> 9
    action_mask: game_history_length -> 9

game_history_t:
    obs:  4       9        3
         ----|----...----|---|
game_history_t+1:
    obs:               4       9       3
                     ----|----...----|---|

game_history_t:
    rew:     9        3  1
         ----...----|---|-|
game_history_t+1:
    rew:             9       3 1
                ----...----|---|-|
"""


one_episode_game_histories = one_episode_replay_buffer_tictocetoe_2_player_mode

print(one_episode_game_histories[0].action_history.shape)
print(one_episode_game_histories[0].obs_history.shape)
print(one_episode_game_histories[0].reward_history.shape)
print(one_episode_game_histories[0].action_mask_history.shape)
print(one_episode_game_histories[0].to_play_history.shape)
print(one_episode_game_histories[0].root_value_history.shape)
print(one_episode_game_histories[0].child_visit_history.shape)
