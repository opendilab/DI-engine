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
from nervex.utils import deep_merge_dicts
from nervex.worker import BaseLearner, BaseSerialCollector, BaseSerialEvaluator, BaseSerialCommander
from nervex.data import BufferManager
from nervex.envs import get_env_cls, get_env_manager_cls
from nervex.policy import get_policy_cls


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


def read_config(cfg: str) -> Tuple[dict, dict]:
    suffix = cfg.split('.')[-1]
    if suffix == 'py':
        cfg = Config.file_to_dict(cfg).cfg_dict
        assert "main_config" in cfg, "Please make sure a 'main_config' variable is declared in config python file!"
        return cfg['main_config'], cfg['create_config']
    else:
        raise KeyError("invalid config file suffix: {}".format(suffix))


def save_config(config_: dict, path: str, type_: str = 'py') -> NoReturn:
    assert type_ in ['yaml', 'py'], type_
    if type_ == 'yaml':
        save_config_yaml(config_, path)
    elif type_ == 'py':
        save_config_py(config_, path)


def deal_with_multi_buffer(default_config: EasyDict, cfg: EasyDict) -> EasyDict:
    if 'buffer_name' in cfg.policy.other.replay_buffer:
        buffer_name = cfg.policy.other.replay_buffer.buffer_name
        single_buffer_default_config = default_config.policy.other.pop('replay_buffer')
        multi_replay_buffer_config = EasyDict({k: copy.deepcopy(single_buffer_default_config) for k in buffer_name})
        multi_replay_buffer_config.buffer_name = buffer_name
        default_config.policy.other.replay_buffer = multi_replay_buffer_config
    return default_config


def compile_config(
    cfg,
    env=None,
    env_manager=None,
    policy=None,
    learner=BaseLearner,
    collector=BaseSerialCollector,
    evaluator=BaseSerialEvaluator,
    buffer=BufferManager,
    auto: bool = False,
    create_cfg: dict = None,
    save_cfg: bool = False,
    save_path: str = 'total_config.py',
):
    if auto:
        assert create_cfg is not None
        if env is None:
            env = get_env_cls(create_cfg.env)
        if env_manager is None:
            env_manager = get_env_manager_cls(create_cfg.env_manager)
        if policy is None:
            policy = get_policy_cls(create_cfg.policy)
        env_config = env.default_config()
        env_config.update(create_cfg.env)
        env_config.manager = env_manager.default_config()
        env_config.manager.update(create_cfg.env_manager)
        policy_config = policy.default_config()
        policy_config.update(create_cfg.policy)
        policy_config.other.commander = BaseSerialCommander.default_config()
    else:
        env_config = env.default_config()
        env_config.manager = env_manager.default_config()
        policy_config = policy.default_config()
    policy_config.learn.learner = learner.default_config()
    policy_config.collect.collector = collector.default_config()
    policy_config.eval.evaluator = evaluator.default_config()
    policy_config.other.replay_buffer = buffer.default_config()
    default_config = EasyDict({'env': env_config, 'policy': policy_config})
    default_config = deal_with_multi_buffer(default_config, cfg)
    cfg = deep_merge_dicts(default_config, cfg)
    # check important key in config
    assert all([k in cfg.env for k in ['n_episode', 'stop_value']]), cfg.env
    if save_cfg:
        save_config(cfg, save_path)
    return cfg
