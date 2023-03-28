from typing import Union, Optional, List, Any, Callable, Tuple
import torch
from ding.config import compile_config, read_config
from ding.envs import get_vec_env_setting
from ding.policy import create_policy
from ding.utils import set_pkg_seed
from ding.torch_utils import to_tensor, to_ndarray, tensor_to_list


def eval(
        input_cfg: Union[str, Tuple[dict, dict]],
        seed: int = 0,
        model: Optional[torch.nn.Module] = None,
        state_dict: Optional[dict] = None,
        replay_path: Optional[str] = './video',
) -> float:
    r"""
    Overview:
        The evaluation entry for NGU policy.
    Arguments:
        - input_cfg (:obj:`Union[str, Tuple[dict, dict]]`): Config in dict type. \
            ``str`` type means config file path. \
            ``Tuple[dict, dict]`` type means [user_config, create_cfg].
        - seed (:obj:`int`): Random seed.
        - env_setting (:obj:`Optional[List[Any]]`): A list with 3 elements: \
            ``BaseEnv`` subclass, collector env config, and evaluator env config.
        - model (:obj:`Optional[torch.nn.Module]`): Instance of torch.nn.Module.
        - state_dict (:obj:`Optional[dict]`): The state_dict of policy or model.
    """
    if isinstance(input_cfg, str):
        cfg, create_cfg = read_config(input_cfg)
    else:
        cfg, create_cfg = input_cfg
    create_cfg.policy.type += '_command'
    cfg = compile_config(cfg, auto=True, create_cfg=create_cfg)
    env_fn, _, evaluator_env_cfg = get_vec_env_setting(cfg.env)
    env = env_fn(evaluator_env_cfg[0])
    env.seed(seed, dynamic_seed=False)
    set_pkg_seed(seed, use_cuda=cfg.policy.cuda)
    policy = create_policy(cfg.policy, model=model, enable_field=['eval']).eval_mode
    if state_dict is None:
        state_dict = torch.load(cfg.learner.load_path, map_location='cpu')
    policy.load_state_dict(state_dict)
    env.enable_save_replay(replay_path=replay_path)
    obs = env.reset()
    obs = {0: obs}
    episode_return = 0.

    beta_index = {i: 0 for i in range(1)}
    beta_index = to_tensor(beta_index, dtype=torch.int64)
    prev_action = {i: torch.tensor(-1) for i in range(1)}
    prev_reward_e = {i: to_tensor(0, dtype=torch.float32) for i in range(1)}

    while True:
        # TODO(pu): r_i, reward embedding
        policy_output = policy.forward(beta_index, obs, prev_action, prev_reward_e)

        actions = {i: a['action'] for i, a in policy_output.items()}
        actions = to_ndarray(actions)

        action = policy_output[0]['action']
        action = to_ndarray(action)
        timestep = env.step(action)
        # print(action)
        # print(timestep.reward)

        timesteps = {0: timestep}
        timesteps = to_tensor(timesteps, dtype=torch.float32)

        prev_reward_e = {env_id: timestep.reward for env_id, timestep in timesteps.items()}
        prev_reward_e = to_ndarray(prev_reward_e)
        prev_action = actions

        timestep = timesteps[0]
        # print(timestep.info)
        episode_return += timestep.reward

        obs = timestep.obs
        obs = {0: obs}

        if timestep.done:
            print(timestep.info)
            break
    print('Eval is over! The performance of your RL policy is {}'.format(episode_return))


if __name__ == "__main__":
    # Users should add their own model path here. Model path should lead to a model.
    # Absolute path is recommended.
    # In DI-engine, it is ``exp_name/ckpt/ckpt_best.pth.tar``.
    model_path = './debug_minigrid_doorkey_ngu_ul298_er01_n32_rbs3e4_fixepseval/ckpt/ckpt_best.pth.tar',
    # model_path = 'model_path_placeholder',
    cfg = '../config/minigrid_ngu_config.py'

    state_dict = torch.load(model_path, map_location='cpu')
    for i in range(0, 10):
        eval(cfg, seed=i, state_dict=state_dict, replay_path='./video')
