from typing import Optional, Callable, Tuple, Dict, List
from collections import namedtuple, defaultdict
import numpy as np
import torch

from ...envs import BaseEnvManager
from ...envs import BaseEnvManager

from ding.envs import BaseEnvManager
from ding.torch_utils import to_tensor, to_ndarray, to_item
from ding.utils import build_logger, EasyTimer, SERIAL_EVALUATOR_REGISTRY
from ding.utils import get_world_size, get_rank, broadcast_object_list
from .base_serial_evaluator import ISerialEvaluator, VectorEvalMonitor
from .interaction_serial_evaluator import InteractionSerialEvaluator

class InteractionSerialMetaEvaluator(InteractionSerialEvaluator):
    """
    Overview:
        Interaction serial evaluator class, policy interacts with env. This class evaluator algorithm
        with test environment list.
    Interfaces:
        __init__, reset, reset_policy, reset_env, close, should_eval, eval
    Property:
        env, policy
    """
    config = dict(
        # (int) Evaluate every "eval_freq" training iterations.
        eval_freq=1000,
        render=dict(
            # Tensorboard video render is disabled by default.
            render_freq=-1,
            mode='train_iter',
        ),
        # (str) File path for visualize environment information.
        figure_path=None,
        # test env list
        test_env_list=None,
    )

    def __init__(
            self,
            cfg: dict,
            env: BaseEnvManager = None,
            policy: namedtuple = None,
            tb_logger: 'SummaryWriter' = None,  # noqa
            exp_name: Optional[str] = 'default_experiment',
            instance_name: Optional[str] = 'evaluator',
    ) -> None:
        super()._init_eval(cfg, env, policy, tb_logger, exp_name, instance_name)
        self.test_env_num = len(cfg.test_env_list)

    def eval(
            self,
            save_ckpt_fn: Callable = None,
            train_iter: int = -1,
            envstep: int = -1,
            n_episode: Optional[int] = None,
            force_render: bool = False,
            policy_kwargs: Optional[Dict] = {},
    ) -> Tuple[bool, Dict[str, List]]:
        top_flags, episode_infos = [], defaultdict(list)
        for i in range(self.test_env_num):
            self._env.reset_task(self._cfg.test_env_list[i])
            top_flag, episode_info = super().eval(save_ckpt_fn, train_iter, envstep, n_episode, \
                                                  force_render, policy_kwargs)
            top_flags.append(top_flag)
            for key, val in episode_info.items():
                if i == 0:
                    episode_infos[key] = []
                episode_infos[key].append(val)
        
        meta_infos = defaultdict(list)
        for key, val in episode_infos.items():
            meta_infos[key] = episode_infos[key].mean()
        return top_flags, meta_infos