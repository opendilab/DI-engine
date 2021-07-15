from dizoo.chess.tic_tac_toe.envs.tic_tac_toe import TicTacToeEnv

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
        env.render()
        if reward > 0:
            print('computer player win')
        else:
            print('draw')
        break