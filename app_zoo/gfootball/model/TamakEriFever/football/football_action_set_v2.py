import importlib
import json
from pathlib import Path
from os import path
import numpy as np
import os
import uuid
import time
import shutil

from gfootball.env import football_action_set


def renderer(state, env):
    html_renderer(env)


def run_right_agent(obs):
    # keep running right.
    return [5] * obs.controlled_players


def run_left_agent(obs):
    # keep running left.
    return [1] * obs.controlled_players


def do_nothing_agent(obs):
    # do nothing.
    return [0] * obs.controlled_players


def builtin_ai_agent(obs):
    # execute builtin AI behavior.
    return [19] * obs.controlled_players


agents = {
    "run_right": run_right_agent,
    "run_left": run_left_agent,
    "do_nothing": do_nothing_agent,
    "builtin_ai": builtin_ai_agent,
}


def parse_single_player(obs_raw_entry):
    # Remove pixel information.
    if "frame" in obs_raw_entry:
        del obs_raw_entry["frame"]
    for k, v in obs_raw_entry.items():
        if type(v) == np.ndarray:
            obs_raw_entry[k] = v.tolist()
    return obs_raw_entry


def try_get_video(env, keep_running=False):
    if not env.football_video_path:
        internal_env = m_envs[env.configuration.id]
        while not hasattr(internal_env, '_env'):
            internal_env = internal_env.env
        if not keep_running and internal_env._env._step == -1:
            # Generate no-op step, so that video is available.
            internal_env.step([0] * (env.configuration.team_1 + env.configuration.team_2))
        trace = internal_env._env._trace
        if trace:
          trace._dump_config['episode_done']._min_frequency = 0
          dumps = trace.process_pending_dumps(True)
          env.football_video_path = retrieve_video_link(dumps)
        if not env.football_video_path:
            return
        if keep_running:
            trace.write_dump('episode_done')
    if 'LiveVideoPath' in env.info and env.info['LiveVideoPath'] is not None:
        target_path = Path(env.info['LiveVideoPath'])
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(env.football_video_path, target_path)
        env.football_video_path = env.info['LiveVideoPath']


def update_observations_and_rewards(configuration, state, obs, rew=None):
    """Updates agent-visible observations given 'raw' observations from environment.
    Observations in 'obs' are coming directly from the environment and are in 'raw' format.
    """
    state[0].observation.controlled_players = configuration.team_1
    state[1].observation.controlled_players = configuration.team_2

    assert len(obs) == configuration.team_1 + configuration.team_2
    if rew is not None:
        state[0].reward = rew
        state[1].reward = -rew
    state[0].observation.players_raw = [
        parse_single_player(obs[x]) for x in range(configuration.team_1)
    ]
    state[1].observation.players_raw = [
        parse_single_player(obs[x + configuration.team_1])
        for x in range(configuration.team_2)
    ]


def mark_invalid(agent, message):
    agent.status = "INVALID"
    agent.reward = -100
    agent.info.debug_info = message


def maybe_terminate(env, state):
    if state[0].status != "ACTIVE" or state[1].status != "ACTIVE":
        if state[0].status == "ACTIVE":
            state[0].status = "DONE"
            state[0].reward = 100
            state[0].info.debug_info = "Opponent forfeited. You win."
        elif not state[0].reward:
            state[0].reward = -100
        if state[1].status == "ACTIVE":
            state[1].status = "DONE"
            state[1].reward = 100
            state[1].info.debug_info = "Opponent forfeited. You win."
        elif not state[1].reward:
            state[1].reward = -100
        try_get_video(env)
        return True
    return False


def football_env():
    # Use lazy-import to avoid this heavy dependency unless it is really needed.
    return importlib.import_module("gfootball.env")


# Global dictionary with active environments.
m_envs = {}


def cleanup(env):
    global m_envs
    del m_envs[env.configuration.id]


def cleanup_all():
    global m_envs
    del m_envs


def retrieve_video_link(dumps):
    for entry in dumps:
        if entry['name'] == 'episode_done':
            print("Received video link.")
            return entry['video']
    return None


def interpreter(state, env):
    global m_envs
    if "id" not in env.configuration or env.configuration.id is None:
        env.configuration.id = str(uuid.uuid4())

    if (env.configuration.id not in m_envs) or env.done:
        if env.configuration.id not in m_envs:
            print("Staring a new environment %s: with scenario: %s" %
                  (env.configuration.id, env.configuration.scenario_name))

            other_config_options = {}
            # Use webm to encode videos (so that you can see them in the browser).
            other_config_options["video_format"] = "webm"
            if env.configuration.running_in_notebook:
                assert not env.configuration.render, "Render is not supported inside notebook environment."

            env.football_video_path = None
            if 'TeamNames' in env.info:
                names = env.info['TeamNames']
                assert len(names) == 2
                other_config_options['custom_display_stats'] = [
                    'LEFT PLAYER: %s' % names[0],
                    'RIGHT PLAYER: %s' % names[1]]
            m_envs[env.configuration.id] = football_env().create_environment(
                env_name=env.configuration.scenario_name,
                stacked=False,
                # We use 'raw' representation to transfer data between server and agents.
                representation='raw',
                logdir=path.join(env.configuration.logdir, env.configuration.id),
                write_goal_dumps=False,
                write_full_episode_dumps=env.configuration.save_video,
                write_video=env.configuration.save_video,
                render=env.configuration.render,
                number_of_left_players_agent_controls=env.configuration.team_1,
                number_of_right_players_agent_controls=env.configuration.team_2,
                other_config_options={**other_config_options, 'action_set': 'v2'})
        else:
            print("Resetting environment %s: with scenario: %s" %
                  (env.configuration.id, env.configuration.scenario_name))
        obs = m_envs[env.configuration.id].reset()
        update_observations_and_rewards(configuration=env.configuration,
                                        state=state,
                                        obs=obs)
    if env.done:
        return state

    if maybe_terminate(env, state):
        return state

    # verify actions.
    controlled_players = env.configuration.team_1
    action_set = football_action_set.action_set_dict['v2']

    try:
        for action in state[0].action:
            football_action_set.named_action_from_action_set(action_set, action)
    except Exception:
        mark_invalid(state[0], "Invalid action provided: %s." % state[0].action)
    if len(state[0].action) != env.configuration.team_1:
        mark_invalid(state[0], "Invalid number of actions provided: Expected %d, got %d." %
            (env.configuration.team_1, len(state[0].action)))
    actions_to_env = state[0].action

    try:
        for action in state[1].action:
            football_action_set.named_action_from_action_set(action_set, action)
    except Exception:
        mark_invalid(state[1], "Invalid action provided: %s." % state[1].action)
    if len(state[1].action) != env.configuration.team_2:
        mark_invalid(state[1], "Invalid number of actions provided: Expected %d, got %d." %
            (env.configuration.team_2, len(state[1].action)))
    if env.configuration.team_2:
        actions_to_env = actions_to_env + state[1].action

    if maybe_terminate(env, state):
        return state
    obs, rew, done, info = m_envs[env.configuration.id].step(actions_to_env)

    if "dumps" in info:
        env.football_video_path = retrieve_video_link(info["dumps"])
    update_observations_and_rewards(configuration=env.configuration,
                                    state=state,
                                    obs=obs,
                                    rew=obs[0]['score'][0]-obs[0]['score'][1])

    ## TODO: pass other information from 'info' to the state/agent.
    if done:
        for agent in range(2):
            state[agent].status = "DONE"
        try_get_video(env)

    return state


dirpath = path.dirname(__file__)
jsonpath = path.abspath(path.join(dirpath, "football.json"))
with open(jsonpath) as f:
    specification = json.load(f)


def html_renderer(env):
    try_get_video(env, keep_running=True)
    if not env.football_video_path:
        raise Exception(
            "No video found. Was environment created with save_video enabled?"
        )

    from IPython.display import display, HTML
    from base64 import b64encode

    video = open(env.football_video_path, 'rb').read()
    env.football_video_path = None
    data_url = "data:video/webm;base64," + b64encode(video).decode()

    html = """
<video width=800 controls>
  <source src="%s" type="video/webm">
</video>
""" % data_url
    display(HTML(html))
    return ""
