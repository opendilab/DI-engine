import pytest
import json
import os
from luxai2021.game.constants import LuxMatchConfigs_Default
from luxai2021.env.lux_env import LuxEnvironment
from luxai2021.env.agent import Agent, AgentFromReplay

@pytest.mark.parametrize("replay_id",['27095556', 
                                      '26835897', 
                                      '26773935', 
                                      '26691974', 
                                      '26688997', 
                                      '26690562', 
                                      '27075871'])
def test_run_replay(replay_id):
    print("Testing simulated replays...")
    print(replay_id)

    script_dir = os.path.dirname(__file__)
    with open(os.path.join(script_dir, f"replays_for_test/{replay_id}.json"), mode="r") as replay_file:
        json_args = json.load(replay_file)
    
    config = LuxMatchConfigs_Default.copy()
    config['seed'] = json_args['configuration']['seed']

    opponent = AgentFromReplay(replay=json_args)
    agent = AgentFromReplay(replay=json_args)

    env = LuxEnvironment(configs=config,
                        learning_agent=agent,
                        opponent_agent=opponent,
                        replay_validate=json_args)

    is_game_error = env.run_no_learn()
    assert not is_game_error

    return True
