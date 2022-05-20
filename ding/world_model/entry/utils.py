from typing import Union, Optional, List, Any, Tuple
import torch
import os
from functools import partial

from tensorboardX import SummaryWriter
from ding.utils.default_helper import deep_merge_dicts

from ding.worker import BaseLearner, InteractionSerialEvaluator, BaseSerialCommander, create_buffer, \
    create_serial_collector
from ding.envs import get_vec_env_setting, create_env_manager
from ding.config import compile_config
from ding.utils import set_pkg_seed
from ding.policy import create_policy
from ding.utils import import_module, WORLD_MODEL_REGISTRY


def get_world_model_cls(cfg):
    import_module(cfg.get('import_names', []))
    return WORLD_MODEL_REGISTRY.get(cfg.type)

def create_world_model(cfg, *args, **kwargs):
    import_module(cfg.get('import_names', []))
    return WORLD_MODEL_REGISTRY.build(cfg.type, cfg, *args, **kwargs)

# TODO: complete the setup with commander and so on
def mbrl_entry_setup(
        input_cfg: Union[str, Tuple[dict, dict]],
        seed: int = 0,
        env_setting: Optional[List[Any]] = None,
        model: Optional[torch.nn.Module] = None,
        max_train_iter: Optional[int] = int(1e10),
        max_env_step: Optional[int] = int(1e10),
):
    cfg, create_cfg = input_cfg
    create_cfg.policy.type = create_cfg.policy.type + '_command'

    cfg = compile_config(cfg, seed=seed, env=None, auto=True, create_cfg=create_cfg, save_cfg=True)
    # merge world model default config
    cfg.world_model = deep_merge_dicts(
        get_world_model_cls(cfg.world_model).default_config(), cfg.world_model)

    env_fn, collector_env_cfg, evaluator_env_cfg = get_vec_env_setting(cfg.env)
    collector_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in collector_env_cfg])
    evaluator_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in evaluator_env_cfg])

    collector_env.seed(cfg.seed)
    evaluator_env.seed(cfg.seed, dynamic_seed=False)

    # create logger
    tb_logger = SummaryWriter(os.path.join('./{}/log/'.format(cfg.exp_name), 'serial'))

    # create world model
    world_model = create_world_model(cfg.world_model, env_fn(cfg.env), tb_logger)

    # create policy
    set_pkg_seed(cfg.seed, use_cuda=cfg.policy.cuda)
    policy = create_policy(cfg.policy, enable_field=['learn', 'collect', 'eval', 'command'])

    # create worker
    learner = BaseLearner(cfg.policy.learn.learner, policy.learn_mode, tb_logger, exp_name=cfg.exp_name)
    collector = create_serial_collector(
        cfg.policy.collect.collector,
        env=collector_env,
        policy=policy.collect_mode,
        tb_logger=tb_logger,
        exp_name=cfg.exp_name
    )
    evaluator = InteractionSerialEvaluator(
        cfg.policy.eval.evaluator, evaluator_env, policy.eval_mode, tb_logger, exp_name=cfg.exp_name
    )
    env_buffer = create_buffer(cfg.policy.other.replay_buffer, tb_logger=tb_logger, exp_name=cfg.exp_name)
    commander = BaseSerialCommander(
        cfg.policy.other.commander, learner, collector, evaluator, env_buffer, policy.command_mode
    )


    return (
        cfg, policy, world_model, 
        env_buffer, learner, collector, collector_env, evaluator, commander,
        tb_logger,
    )
