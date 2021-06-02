import json
import os.path as osp
import shutil
import sys
import tempfile
import copy
from importlib import import_module
from typing import Optional, Tuple, NoReturn

import yaml
from easydict import EasyDict
from nervex import policy
from nervex.utils import deep_merge_dicts
from nervex.worker import BaseLearner, SampleCollector, BaseSerialEvaluator, BaseSerialCommander, Coordinator, \
    get_parallel_commander_cls, get_parallel_collector_cls, get_buffer_cls, get_serial_collector_cls, replay_buffer
from nervex.envs import get_env_cls, get_env_manager_cls
from nervex.policy import get_policy_cls, policy_factory
from .utils import parallel_transform, parallel_transform_slurm


class Config(object):

    def __init__(
            self,
            cfg_dict: Optional[dict] = None,
            cfg_text: Optional[str] = None,
            filename: Optional[str] = None
    ) -> None:
        if cfg_dict is None:
            cfg_dict = {}
        if not isinstance(cfg_dict, dict):
            raise TypeError("invalid type for cfg_dict: {}".format(type(cfg_dict)))
        self._cfg_dict = cfg_dict
        if cfg_text:
            text = cfg_text
        elif filename:
            with open(filename, 'r') as f:
                text = f.read()
        else:
            text = '.'
        self._text = text
        self._filename = filename

    @staticmethod
    def file_to_dict(filename: str) -> 'Config':  # noqa
        cfg_dict, cfg_text = Config._file_to_dict(filename)
        return Config(cfg_dict, cfg_text, filename=filename)

    @staticmethod
    def _file_to_dict(filename: str) -> Tuple[dict, str]:
        filename = osp.abspath(osp.expanduser(filename))
        # TODO check exist
        # TODO check suffix
        ext_name = osp.splitext(filename)[-1]
        with tempfile.TemporaryDirectory() as temp_config_dir:
            temp_config_file = tempfile.NamedTemporaryFile(dir=temp_config_dir, suffix=ext_name)
            temp_config_name = osp.basename(temp_config_file.name)
            shutil.copyfile(filename, temp_config_file.name)

            temp_module_name = osp.splitext(temp_config_name)[0]
            sys.path.insert(0, temp_config_dir)
            # TODO validate py syntax
            module = import_module(temp_module_name)
            cfg_dict = {k: v for k, v in module.__dict__.items() if not k.startswith('_')}
            del sys.modules[temp_module_name]
            sys.path.pop(0)
            temp_config_file.close()

        cfg_text = filename + '\n'
        with open(filename, 'r') as f:
            cfg_text += f.read()

        return cfg_dict, cfg_text

    @property
    def cfg_dict(self) -> dict:
        return self._cfg_dict


def read_config_yaml(path: str) -> EasyDict:
    """
    Overview:
        read configuration from path
    Arguments:
        - path (:obj:`str`): Path of source yaml
    Returns:
        - (:obj:`EasyDict`): Config data from this file with dict type
    """
    with open(path, "r") as f:
        config_ = yaml.safe_load(f)

    return EasyDict(config_)


def save_config_yaml(config_: dict, path: str) -> NoReturn:
    """
    Overview:
        save configuration to path
    Arguments:
        - config (:obj:`dict`): Config data
        - path (:obj:`str`): Path of target yaml
    """
    config_string = json.dumps(config_)
    with open(path, "w") as f:
        yaml.safe_dump(json.loads(config_string), f)


def save_config_py(config_: dict, path: str) -> NoReturn:
    """
    Overview:
        save configuration to python file
    Arguments:
        - config (:obj:`dict`): Config data
        - path (:obj:`str`): Path of target yaml
    """
    # config_string = json.dumps(config_, indent=4)
    config_string = str(config_)
    from yapf.yapflib.yapf_api import FormatCode
    config_string, _ = FormatCode(config_string)
    config_string = config_string.replace('inf', 'float("inf")')
    with open(path, "w") as f:
        f.write('exp_config=' + config_string)


def read_config(cfg: str, direct=False) -> Tuple[dict, dict]:
    suffix = cfg.split('.')[-1]
    if suffix == 'py':
        cfg = Config.file_to_dict(cfg).cfg_dict
        if direct:
            return cfg
        assert "main_config" in cfg, "Please make sure a 'main_config' variable is declared in config python file!"
        if 'system_config' in cfg:
            return cfg['main_config'], cfg['create_config'], cfg['system_config']
        else:
            return cfg['main_config'], cfg['create_config']
    else:
        raise KeyError("invalid config file suffix: {}".format(suffix))


def save_config(config_: dict, path: str, type_: str = 'py') -> NoReturn:
    assert type_ in ['yaml', 'py'], type_
    if type_ == 'yaml':
        save_config_yaml(config_, path)
    elif type_ == 'py':
        save_config_py(config_, path)


def compile_buffer_config(policy_cfg: EasyDict, user_cfg: EasyDict, buffer: 'IBuffer') -> EasyDict:  # noqa

    def _compile_buffer_config(policy_buffer_type, user_buffer_type, buffer):
        if user_buffer_type is None or user_buffer_type == policy_buffer_type:
            if buffer is None:
                buffer_cls = get_buffer_cls(policy_buffer_type)
            else:
                buffer_cls = buffer
            return deep_merge_dicts(buffer_cls.default_config(), policy_cfg.other.replay_buffer)
        else:
            if buffer is None:
                buffer_cls = get_buffer_cls(user_buffer_type)
            else:
                buffer_cls = buffer
            return buffer_cls.default_config()

    policy_multi_buffer = policy_cfg.other.replay_buffer.get('multi_buffer', False)
    user_multi_buffer = user_cfg.policy.get('policy', {}).get('other', {}).get('replay_buffer',
                                                                               {}).get('multi_buffer', None)
    assert user_multi_buffer is None or user_multi_buffer == policy_multi_buffer, "For multi_buffer, user_cfg({}) and policy_cfg({}) must be in accordance".format(
        user_multi_buffer, policy_multi_buffer
    )
    multi_buffer = policy_multi_buffer
    if not multi_buffer:
        policy_buffer_type = policy_cfg.other.replay_buffer.type
        user_buffer_type = user_cfg.get('policy', {}).get('other', {}).get('replay_buffer', {}).get('type', None)
        return _compile_buffer_config(policy_buffer_type, user_buffer_type, buffer)
    else:
        return_cfg = EasyDict()
        for buffer_name in policy_cfg.other.replay_buffer:  # Only traverse keys in policy_cfg
            if buffer_name == 'multi_buffer':
                continue
            policy_buffer_type = policy_cfg.other.replay_buffer[buffer_name].type
            user_buffer_type = user_cfg.get('policy', {}).get('other', {}).get('replay_buffer',
                                                                               {}).get('buffer_name',
                                                                                       {}).get('type', None)
            return_cfg[buffer_name] = _compile_buffer_config(policy_buffer_type, user_buffer_type, buffer)
            return_cfg[buffer_name].name = buffer_name
        return return_cfg


def compile_collector_config(
        policy_cfg: EasyDict, user_cfg: EasyDict, collector: 'ISerialCollector'
) -> EasyDict:  # noqa
    policy_collector_type = policy_cfg.collect.collector.type
    user_collector_type = user_cfg.get('policy', {}).get('collect', {}).get('collector', {}).get('type', None)
    if user_collector_type is None or user_collector_type == policy_collector_type:
        if collector is None:
            collector_cls = get_serial_collector_cls(policy_collector_type)
        else:
            collector_cls = collector
        return deep_merge_dicts(collector_cls.default_config(), policy_cfg.collect.collector)
    else:
        if collector is None:
            collector_cls = get_serial_collector_cls(user_collector_type)
        else:
            collector_cls = collector
        return collector_cls.default_config()


def compile_config(
        cfg,
        env_manager=None,
        policy=None,
        learner=BaseLearner,
        collector=None,
        evaluator=BaseSerialEvaluator,
        buffer=None,
        env=None,
        seed: int = 0,
        auto: bool = False,
        create_cfg: dict = None,
        save_cfg: bool = True,
        save_path: str = 'total_config.py',
) -> EasyDict:
    if auto:
        assert create_cfg is not None
        if env is None:
            env = get_env_cls(create_cfg.env)
        if env_manager is None:
            env_manager = get_env_manager_cls(create_cfg.env_manager)
        if policy is None:
            policy = get_policy_cls(create_cfg.policy)
        if 'default_config' in dir(env):
            env_config = env.default_config()
        else:
            env_config = EasyDict()  # env does not have default_config
        env_config.update(create_cfg.env)
        env_config.manager = deep_merge_dicts(env_manager.default_config(), env_config.get('manager', EasyDict()))
        env_config.manager.update(create_cfg.env_manager)
        policy_config = policy.default_config()
        policy_config.update(create_cfg.policy)
        policy_config.other.commander = BaseSerialCommander.default_config()
    else:
        if 'default_config' in dir(env):
            env_config = env.default_config()
        else:
            env_config = EasyDict()  # env does not have default_config
        env_config.manager = deep_merge_dicts(env_manager.default_config(), env_config.get('manager', EasyDict()))
        policy_config = policy.default_config()
    policy_config.learn.learner = deep_merge_dicts(
        learner.default_config(), policy_config.learn.get('learner', EasyDict())
    )
    policy_config.collect.collector = compile_collector_config(policy_config, cfg, collector)
    policy_config.eval.evaluator = deep_merge_dicts(
        evaluator.default_config(), policy_config.eval.get('evaluator', EasyDict())
    )
    policy_config.other.replay_buffer = compile_buffer_config(policy_config, cfg, buffer)
    print(policy_config.other.replay_buffer)
    default_config = EasyDict({'env': env_config, 'policy': policy_config})
    cfg = deep_merge_dicts(default_config, cfg)
    print(cfg.policy.other.replay_buffer)
    cfg.seed = seed
    # check important key in config
    assert all([k in cfg.env for k in ['n_evaluator_episode', 'stop_value']]), cfg.env
    cfg.policy.eval.evaluator.stop_value = cfg.env.stop_value
    cfg.policy.eval.evaluator.n_episode = cfg.env.n_evaluator_episode
    if save_cfg:
        save_config(cfg, save_path)
    return cfg


def compile_config_parallel(
        cfg: EasyDict,
        create_cfg: EasyDict,
        system_cfg: EasyDict,
        seed: int = 0,
        save_cfg: bool = True,
        save_path: str = 'total_config.py',
        platform: str = 'local',
        coordinator_host: Optional[str] = None,
        learner_host: Optional[str] = None,
        collector_host: Optional[str] = None,
) -> EasyDict:
    # get cls
    env = get_env_cls(create_cfg.env)
    if 'default_config' in dir(env):
        env_config = env.default_config()
    else:
        env_config = EasyDict()  # env does not have default_config
    policy = get_policy_cls(create_cfg.policy)
    env_manager = get_env_manager_cls(create_cfg.env_manager)
    env_config.manager = env_manager.default_config()
    policy_config = policy.default_config()
    collector = get_parallel_collector_cls(create_cfg.collector)
    policy_config.collect.collector = collector.default_config()
    policy_config.learn.learner = BaseLearner.default_config()
    commander = get_parallel_commander_cls(create_cfg.commander)
    policy_config.other.commander = commander.default_config()
    policy_config.other.replay_buffer = compile_buffer_config(policy_config, cfg, None)
    default_config = EasyDict({'env': env_config, 'policy': policy_config})

    cfg.env.update(create_cfg.env)
    if 'manager' not in cfg.env:
        cfg.env.manager = {}
    cfg.env.manager.update(create_cfg.env_manager)
    cfg.policy.update(create_cfg.policy)
    cfg.policy.learn.learner.update(create_cfg.learner)
    cfg.policy.collect.collector.update(create_cfg.collector)
    cfg.policy.other.commander.update(create_cfg.commander)
    cfg = deep_merge_dicts(default_config, cfg)

    cfg.policy.other.commander.path_policy = system_cfg.path_policy  # league may use 'path_policy'

    for k in ['comm_learner', 'comm_collector']:
        system_cfg[k] = create_cfg[k]
    if platform == 'local':
        cfg = parallel_transform(EasyDict({'main': cfg, 'system': system_cfg}))
    elif platform == 'slurm':
        cfg = parallel_transform_slurm(
            EasyDict({
                'main': cfg,
                'system': system_cfg
            }), coordinator_host, learner_host, collector_host
        )
    else:
        raise KeyError("not support platform type: {}".format(platform))
    cfg.seed = seed

    cfg.system.coordinator = deep_merge_dicts(Coordinator.default_config(), cfg.system.coordinator)
    if save_cfg:
        save_config(cfg, save_path)
    return cfg
