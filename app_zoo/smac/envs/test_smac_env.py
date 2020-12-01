import numpy as np

from sc2learner.envs.smac.smac_env import SMACEnv

MOVE_EAST = 4
MOVE_WEST = 5


def automation(env, n_agents):
    actions = {"me": [], "opponent": []}
    for agent_id in range(n_agents):
        avail_actions = env.get_avail_agent_actions(agent_id, is_opponent=False)
        avail_actions_ind = np.nonzero(avail_actions)[0]
        action = np.random.choice(avail_actions_ind)
        if MOVE_EAST in avail_actions_ind:
            action = MOVE_EAST
        # Let OPPONENT attack ME at the first place
        if sum(avail_actions[6:]) > 0:
            action = max(avail_actions_ind)
            # print("ME start attacking OP")
        # print("Available action for ME: ", avail_actions_ind)
        actions["me"].append(action)
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


def main(policy, map_name="3m", two_player=True):
    env = SMACEnv(two_player=two_player, map_name=map_name)
    n_agents = 8 if map_name == "3s5z" else 3
    n_episodes = 1000
    me_win = 0
    draw = 0
    op_win = 0

    for e in range(n_episodes):
        print("Now reset the environment for {} episode.".format(e))
        env.reset()
        terminated = False
        episode_reward_me = 0
        episode_reward_op = 0

        while not terminated:
            obs = env.get_obs()
            obs_opponent = env.get_obs(is_opponent=True)
            state = env.get_state()
            actions = policy(env, n_agents)
            if not two_player:
                actions = actions["me"]
            t = env.step(actions)
            reward, terminated, infos = t.reward, t.done, t.info
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


if __name__ == '__main__':
    # main(random_policy, wrap=True)
    main(automation, map_name="3s5z", two_player=False)
    # main(automation, wrap=True)
    # main(lambda a, b: fix_policy(a, b, me=MOVE_WEST, opponent=MOVE_WEST), wrap=True)
