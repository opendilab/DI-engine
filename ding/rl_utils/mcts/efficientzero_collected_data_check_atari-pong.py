import numpy as np
import sys

sys.path.extend(['/Users/puyuan/code/DI-engine', '/Users/puyuan/code/EfficientZero'])

one_episode_replay_buffer_img_ez = np.load(
    '/Users/puyuan/code/DI-engine/dizoo/board_games/atari/config/one_episode_replay_buffer_img_ez.npy', allow_pickle=True
)

one_episode_replay_buffer_img = np.load(
    '/Users/puyuan/code/DI-engine/dizoo/board_games/atari/config/one_episode_replay_buffer_img.npy', allow_pickle=True
)

"""
Note: please refer to ding/rl_utils/mcts/game.py for details.

game_history element shape:
    e.g. game_history_length=20, stack=4, num_unroll_steps=5, td_steps=5

    obs: game_history_length + stack + num_unroll_steps, 20+4 +5
    action: game_history_length -> 20
    reward: game_history_length  + num_unroll_steps + td_steps -1  20 +5+5-1
    root_values:  game_history_length + num_unroll_steps + td_steps -> 20 +5+5
    child_visits： game_history_length + num_unroll_steps -> 20 +5
    to_play: game_history_length -> 20
    action_mask: game_history_length -> 20

game_history_t:
    obs:  4       20        5
         ----|----...----|-----|
game_history_t+1:
    obs:               4       20        5
                     ----|----...----|-----|

game_history_t:
    rew:     20        5      4
         ----...----|------|-----|
game_history_t+1:
    rew:             20        5    4
                ----...----|-----|-----|
"""

print('the total size of one frame is:', 96 * 96 * 3)
for i in range(min(one_episode_replay_buffer_img_ez[0].obs_history.shape[0],
                   one_episode_replay_buffer_img[0].obs_history.shape[0]) - 4):
    print(
        f'the difference in the {i}th frame:',
        (one_episode_replay_buffer_img_ez[0].obs_history[4 + i] -
         one_episode_replay_buffer_img[0].obs_history[4 + i]).sum()
    )

for one_episode_game_histories in [one_episode_replay_buffer_img_ez, one_episode_replay_buffer_img]:

    for i in range(3):
        assert (one_episode_game_histories[0].obs_history[i] -
                one_episode_replay_buffer_img[0].obs_history[i]).sum() == 0

    print('the total size of one frame is:', 96 * 96 * 3)
    print(
        'the difference in the first frame:',
        (one_episode_game_histories[0].obs_history[4] - one_episode_replay_buffer_img[0].obs_history[4]).sum()
    )
    print(
        'the difference in the 2nd frame:',
        (one_episode_game_histories[0].obs_history[5] - one_episode_replay_buffer_img[0].obs_history[5]).sum()
    )

    print(one_episode_game_histories[0].obs_history.shape)
    print(one_episode_game_histories[0].reward_history.shape)

    all_same = np.array(one_episode_game_histories[0].obs_history[0] == one_episode_game_histories[0].obs_history[0]
                        ).astype(int).sum()
    """
    check rewards in the neighboring game history
    """
    for i in range(13):
        print(
            np.array(one_episode_game_histories[0].reward_history[16 + i] == one_episode_game_histories[1].reward_history[i]
                     ).astype(int)
        )
    """
    check obs_history in the neighboring game history
    """
    for i in range(9):
        print(
            np.array(one_episode_game_histories[0].obs_history[20 + i] == one_episode_game_histories[1].obs_history[i]
                     ).astype(int).sum() == all_same
        )
