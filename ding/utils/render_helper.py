def render(env, render_mode=('rgb_array')):
    '''
    Overview:
        Render the environment's current frame.
    Arguments:
        - env (:obj:`BaseEnv`): the ding env.
        - render_mode (:obj:`str`): render mode.
    Returns:
        - frame (:obj:`numpy.ndarray`): [H * W * C]
    '''
    # TODO: do not use private member _env
    if hasattr(env._env, 'sim'):
        # mujoco
        return env._env.sim.render(camera_name='track', height=128, width=128)[::-1]
    else:
        # other
        return env._env.render(mode=render_mode)

def fps(env_manager):
    '''
    Overview:
        Render the environment's fps.
    Arguments:
        - env (:obj:`BaseEnvManager`): the ding env manager.
    Returns:
        - fps (:obj:`int`).
    '''
    # TODO: do not use private member _env_ref
    env_ref = env_manager._env_ref
    if hasattr(env_ref, 'model'):
        # mujoco
        fps = 1 / env_ref.model.opt.timestep
    elif hasattr(env_ref, 'env') and 'video.frames_per_second' in env_ref.env.metadata.keys():
        # classic control
        fps = env_ref.env.metadata['video.frames_per_second']
    else:
        # atari and others
        fps = 30
    return fps
