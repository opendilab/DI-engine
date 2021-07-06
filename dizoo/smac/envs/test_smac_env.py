import pytest
import numpy as np
from easydict import EasyDict

from dizoo.smac.envs import SMACEnv

MOVE_EAST = 4
MOVE_WEST = 5


def automation(env, n_agents):
    actions = {"me": [], "opponent": []}
    for agent_id in range(n_agents):
        avail_actions = env.get_avail_agent_actions(agent_id, is_opponent=False)
        avail_actions_ind = np.nonzero(avail_actions)[0]
        action = np.random.choice(avail_actions_ind)
        if avail_actions[0] != 0:
            action = 0
        elif len(np.nonzero(avail_actions[6:])[0]) == 0:
            if avail_actions[MOVE_EAST] != 0:
                action = MOVE_EAST
            else:
                action = np.random.choice(avail_actions_ind)
        else:
            action = np.random.choice(avail_actions_ind)
        # if MOVE_EAST in avail_actions_ind:
        #     action = MOVE_EAST
        # Let OPPONENT attack ME at the first place
        # if sum(avail_actions[6:]) > 0:
        #     action = max(avail_actions_ind)
        # print("ME start attacking OP")
        # print("Available action for ME: ", avail_actions_ind)
        actions["me"].append(action)
        print('ava', avail_actions, action)
    for agent_id in range(n_agents):
        avail_actions = env.get_avail_agent_actions(agent_id, is_opponent=True)
        avail_actions_ind = np.nonzero(avail_actions)[0]
        action = np.random.choice(avail_actions_ind)
        if MOVE_EAST in avail_actions_ind:
            action = MOVE_EAST
        # Let OPPONENT attack ME at the first place
        if sum(avail_actions[6:]) > 0:
            # print("OP start attacking ME")
            action = max(avail_actions_ind)
        actions["opponent"].append(action)
    return actions


def random_policy(env, n_agents):
    actions = {"me": [], "opponent": []}
    for agent_id in range(n_agents):
        avail_actions = env.get_avail_agent_actions(agent_id, is_opponent=False)
        avail_actions_ind = np.nonzero(avail_actions)[0]
        action = np.random.choice(avail_actions_ind)
        actions["me"].append(action)
    for agent_id in range(n_agents):
        avail_actions = env.get_avail_agent_actions(agent_id, is_opponent=True)
        avail_actions_ind = np.nonzero(avail_actions)[0]
        # Move left to kill ME
        action = np.random.choice(avail_actions_ind)
        actions["opponent"].append(action)
    return actions


def fix_policy(env, n_agents, me=0, opponent=0):
    actions = {"me": [], "opponent": []}
    for agent_id in range(n_agents):
        avail_actions = env.get_avail_agent_actions(agent_id, is_opponent=False)
        avail_actions_ind = np.nonzero(avail_actions)[0]
        action = me
        if action not in avail_actions_ind:
            action = avail_actions_ind[0]
        actions["me"].append(action)

    for agent_id in range(n_agents):
        avail_actions = env.get_avail_agent_actions(agent_id, is_opponent=True)
        avail_actions_ind = np.nonzero(avail_actions)[0]
        action = opponent
        if action not in avail_actions_ind:
            action = avail_actions_ind[0]
        actions["opponent"].append(action)
    return actions


def main(policy, map_name="3m", two_player=False):
    cfg = EasyDict({'two_player': two_player, 'map_name': map_name, 'save_replay_episodes': None, 'obs_alone': True})
    env = SMACEnv(cfg)
    if map_name == "3s5z":
        n_agents = 8
    elif map_name == "3m":
        n_agents = 3
    elif map_name == "infestor_viper":
        n_agents = 2
    else:
        raise ValueError(f"invalid type: {map_name}")
    n_episodes = 20
    me_win = 0
    draw = 0
    op_win = 0

    for e in range(n_episodes):
        print("Now reset the environment for {} episode.".format(e))
        env.reset()
        print('reset over')
        terminated = False
        episode_reward_me = 0
        episode_reward_op = 0

        env_info = env.info()
        print('begin new episode')
        while not terminated:
            actions = policy(env, n_agents)
            if not two_player:
                actions = actions["me"]
            t = env.step(actions)
            obs, reward, terminated, infos = t.obs, t.reward, t.done, t.info
            assert set(obs.keys()) == set(
                ['agent_state', 'global_state', 'action_mask', 'agent_alone_state', 'agent_alone_padding_state']
            )
            assert isinstance(obs['agent_state'], np.ndarray)
            assert obs['agent_state'].shape == env_info.obs_space.shape['agent_state']  # n_agents, agent_state_dim
            assert isinstance(obs['agent_alone_state'], np.ndarray)
            assert obs['agent_alone_state'].shape == env_info.obs_space.shape['agent_alone_state']
            assert isinstance(obs['global_state'], np.ndarray)
            assert obs['global_state'].shape == env_info.obs_space.shape['global_state']  # global_state_dim
            assert isinstance(reward, np.ndarray)
            assert reward.shape == (1, )
            print('reward', reward)
            assert isinstance(terminated, bool)
            episode_reward_me += reward["me"] if two_player else reward
            episode_reward_op += reward["opponent"] if two_player else 0
            terminated = terminated["me"] if two_player else terminated

        if two_player:
            me_win += int(infos["me"]["battle_won"])
            op_win += int(infos["opponent"]["battle_won"])
            draw += int(infos["draw"])
        else:
            me_win += int(infos["battle_won"])
            op_win += int(infos["battle_lost"])
            draw += int(infos["draw"])

        print(
            "Total reward in episode {} = {} (me), {} (opponent). Me win {}, Draw {}, Opponent win {}, total {}."
            "".format(e, episode_reward_me, episode_reward_op, me_win, draw, op_win, e + 1)
        )

    env.close()


@pytest.mark.env_test
def test_automation():
    # main(automation, map_name="3m", two_player=False)
    main(automation, map_name="infestor_viper", two_player=False)


if __name__ == "__main__":
    test_automation()
