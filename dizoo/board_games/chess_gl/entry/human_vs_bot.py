from dizoo.board_games.chess_gl.envs.chess_env import ChessAlphaZeroV0Env


env = ChessAlphaZeroV0Env()
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
    action = env.random_action()
    print('computer player ' + env.action_to_string(action))
    obs, reward, done, info = env.step(action)
    if done:
        if reward > 0:
            print('computer player win')
        else:
            print('draw')
        break