from typing import Union, Optional, List, Any, Tuple
import os
import torch
from functools import partial
from tensorboardX import SummaryWriter
from copy import deepcopy
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from ding.envs import get_vec_env_setting, create_env_manager
from ding.worker import BaseLearner, InteractionSerialEvaluator, create_serial_evaluator
from ding.config import read_config, compile_config
from ding.policy import create_policy
from ding.utils import set_pkg_seed, get_world_size, get_rank
from ding.utils.data import create_dataset

from qtransformer.algorithm.dataset_qtransformer import ReplayMemoryDataset
import wandb
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Union

from easydict import EasyDict


def merge_dict1_into_dict2(
    dict1: Union[Dict, EasyDict], dict2: Union[Dict, EasyDict]
) -> Union[Dict, EasyDict]:
    """
    Overview:
        Merge two dictionaries recursively. \
        Update values in dict2 with values in dict1, and add new keys from dict1 to dict2.
    Arguments:
        - dict1 (:obj:`dict`): The first dictionary.
        - dict2 (:obj:`dict`): The second dictionary.
    """
    for key, value in dict1.items():
        if key in dict2 and isinstance(value, dict) and isinstance(dict2[key], dict):
            # Both values are dictionaries, so merge them recursively
            merge_dict1_into_dict2(value, dict2[key])
        else:
            # Either the key doesn't exist in dict2 or the values are not dictionaries
            dict2[key] = value

    return dict2


def merge_two_dicts_into_newone(
    dict1: Union[Dict, EasyDict], dict2: Union[Dict, EasyDict]
) -> Union[Dict, EasyDict]:
    """
    Overview:
        Merge two dictionaries recursively into a new dictionary. \
        Update values in dict2 with values in dict1, and add new keys from dict1 to dict2.
    Arguments:
        - dict1 (:obj:`dict`): The first dictionary.
        - dict2 (:obj:`dict`): The second dictionary.
    """
    dict2 = deepcopy(dict2)
    return merge_dict1_into_dict2(dict1, dict2)


def serial_pipeline_offline(
    input_cfg: Union[str, Tuple[dict, dict]],
    seed: int = 0,
    env_setting: Optional[List[Any]] = None,
    model: Optional[torch.nn.Module] = None,
    max_train_iter: Optional[int] = int(1e10),
) -> "Policy":  # noqa
    """
    Overview:
        Serial pipeline entry.
    Arguments:
        - input_cfg (:obj:`Union[str, Tuple[dict, dict]]`): Config in dict type. \
            ``str`` type means config file path. \
            ``Tuple[dict, dict]`` type means [user_config, create_cfg].
        - seed (:obj:`int`): Random seed.
        - env_setting (:obj:`Optional[List[Any]]`): A list with 3 elements: \
            ``BaseEnv`` subclass, collector env config, and evaluator env config.
        - model (:obj:`Optional[torch.nn.Module]`): Instance of torch.nn.Module.
        - max_train_iter (:obj:`Optional[int]`): Maximum policy update iterations in training.
    Returns:
        - policy (:obj:`Policy`): Converged policy.
    """
    if isinstance(input_cfg, str):
        cfg, create_cfg = read_config(input_cfg)
    else:
        cfg, create_cfg = deepcopy(input_cfg)
    create_cfg.policy.type = create_cfg.policy.type + "_command"
    cfg = compile_config(cfg, seed=seed, auto=True, create_cfg=create_cfg)

    # Dataset
    dataloader = DataLoader(
        ReplayMemoryDataset(**cfg.dataset),
        batch_size=cfg.policy.learn.batch_size,
        shuffle=True,
    )

    env_fn, _, evaluator_env_cfg = get_vec_env_setting(cfg.env, collect=False)
    evaluator_env = create_env_manager(
        cfg.env.manager, [partial(env_fn, cfg=c) for c in evaluator_env_cfg]
    )
    # Random seed
    evaluator_env.seed(cfg.seed, dynamic_seed=False)
    set_pkg_seed(cfg.seed, use_cuda=cfg.policy.cuda)

    # here
    policy = create_policy(cfg.policy, model=model, enable_field=["learn", "eval"])

    wandb.init(**cfg.wandb)
    config = merge_two_dicts_into_newone(EasyDict(wandb.config), cfg)
    wandb.config.update(config)
    tb_logger = SummaryWriter(os.path.join("./{}/log/".format(cfg.exp_name), "serial"))
    learner = BaseLearner(
        cfg.policy.learn.learner, policy.learn_mode, tb_logger, exp_name=cfg.exp_name
    )
    evaluator = create_serial_evaluator(
        cfg.policy.eval.evaluator,
        env=evaluator_env,
        policy=policy.eval_mode,
        tb_logger=tb_logger,
        exp_name=cfg.exp_name,
    )
    # ==========
    # Main loop
    # ==========
    # Learner's before_run hook.
    learner.call_hook("before_run")
    stop = False

    for epoch in range(cfg.policy.learn.train_epoch):
        if get_world_size() > 1:
            dataloader.sampler.set_epoch(epoch)
        for train_data in dataloader:
            learner.train(train_data)
        if evaluator.should_eval(learner.train_iter):
            stop, eval_info = evaluator.eval(
                learner.save_checkpoint, learner.train_iter
            )
            import numpy as np

            mean_value = np.mean(eval_info["eval_episode_return"])
            std_value = np.std(eval_info["eval_episode_return"])
            max_value = np.max(eval_info["eval_episode_return"])
            wandb.log(
                {"mean": mean_value, "std": std_value, "max": max_value}, commit=False
            )
        if stop or learner.train_iter >= max_train_iter:
            stop = True
            break

    learner.call_hook("after_run")
    if get_rank() == 0:
        import time
        import pickle
        import numpy as np

        with open(os.path.join(cfg.exp_name, "result.pkl"), "wb") as f:
            eval_value_raw = eval_info["eval_episode_return"]
            final_data = {
                "stop": stop,
                "train_iter": learner.train_iter,
                "eval_value": np.mean(eval_value_raw),
                "eval_value_raw": eval_value_raw,
                "finish_time": time.ctime(),
            }
            pickle.dump(final_data, f)
    return policy
