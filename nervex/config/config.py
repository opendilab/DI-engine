import json
import os.path as osp
import shutil
import sys
import tempfile
from importlib import import_module
from typing import Optional, Tuple, NoReturn
import yaml
from easydict import EasyDict

from nervex.utils import deep_merge_dicts
from nervex.envs import get_env_cls, get_env_manager_cls
from nervex.policy import get_policy_cls
from nervex.worker import BaseLearner, BaseSerialEvaluator, BaseSerialCommander, Coordinator, \
    get_parallel_commander_cls, get_parallel_collector_cls, get_buffer_cls, get_serial_collector_cls
from .utils import parallel_transform, parallel_transform_slurm, parallel_transform_k8s


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


def save_config(config_: dict, path: str, type_: str = 'py', save_formatted: bool = False) -> NoReturn:
    assert type_ in ['yaml', 'py'], type_
    if type_ == 'yaml':
        save_config_yaml(config_, path)
    elif type_ == 'py':
        save_config_py(config_, path)
        if save_formatted:
            formated_path = osp.join(osp.dirname(path), 'formatted_' + osp.basename(path))
            save_config_formatted(config_, formated_path)


def save_config_formatted(config_: dict, path: str = 'formatted_total_config.py') -> NoReturn:
    with open(path, "w") as f:
        f.write('from easydict import EasyDict\n\n')
        f.write('main_config = dict(\n')
        for k, v in config_.items():
            if (k == 'env'):
                f.write('    env=dict(\n')
                for k2, v2 in v.items():
                    if (k2 != 'type' and k2 != 'import_names' and k2 != 'manager'):
                        if (isinstance(v2, str)):
                            f.write("        {}='{}',\n".format(k2, v2))
                        else:
                            f.write("        {}={},\n".format(k2, v2))
                    if (k2 == 'manager'):
                        f.write("        manager=dict(\n")
                        for k3, v3 in v2.items():
                            if (v3 != 'cfg_type' and v3 != 'type'):
                                if (isinstance(v3, str)):
                                    f.write("            {}='{}',\n".format(k3, v3))
                                elif v3 == float('inf'):
                                    f.write("            {}=float('{}'),\n".format(k3, v3))
                                else:
                                    f.write("            {}={},\n".format(k3, v3))
                        f.write("        ),\n")
                f.write("    ),\n")
            if (k == 'policy'):
                f.write('    policy=dict(\n')
                for k2, v2 in v.items():
                    if (k2 != 'type' and k2 != 'learn' and k2 != 'collect' and k2 != 'eval' and k2 != 'other'
                            and k2 != 'model'):
                        if (isinstance(v2, str)):
                            f.write("        {}='{}',\n".format(k2, v2))
                        else:
                            f.write("        {}={},\n".format(k2, v2))
                    elif (k2 == 'learn'):
                        f.write("        learn=dict(\n")
                        for k3, v3 in v2.items():
                            if (k3 != 'learner'):
                                if (isinstance(v3, str)):
                                    f.write("            {}='{}',\n".format(k3, v3))
                                else:
                                    f.write("            {}={},\n".format(k3, v3))
                            if (k3 == 'learner'):
                                f.write("            learner=dict(\n")
                                for k4, v4 in v3.items():
                                    if (k4 != 'dataloader' and k4 != 'hook'):
                                        if (isinstance(v4, str)):
                                            f.write("                {}='{}',\n".format(k4, v4))
                                        else:
                                            f.write("                {}={},\n".format(k4, v4))
                                    else:
                                        if (k4 == 'dataloader'):
                                            f.write("                dataloader=dict(\n")
                                            for k5, v5 in v4.items():
                                                if (isinstance(v5, str)):
                                                    f.write("                    {}='{}',\n".format(k5, v5))
                                                else:
                                                    f.write("                    {}={},\n".format(k5, v5))
                                            f.write("                ),\n")
                                        if (k4 == 'hook'):
                                            f.write("                hook=dict(\n")
                                            for k5, v5 in v4.items():
                                                if (isinstance(v5, str)):
                                                    f.write("                    {}='{}',\n".format(k5, v5))
                                                else:
                                                    f.write("                    {}={},\n".format(k5, v5))
                                            f.write("                ),\n")
                                f.write("            ),\n")
                        f.write("        ),\n")
                    elif (k2 == 'collect'):
                        f.write("        collect=dict(\n")
                        for k3, v3 in v2.items():
                            if (k3 != 'collector'):
                                if (isinstance(v3, str)):
                                    f.write("            {}='{}',\n".format(k3, v3))
                                else:
                                    f.write("            {}={},\n".format(k3, v3))
                            if (k3 == 'collector'):
                                f.write("            collector=dict(\n")
                                for k4, v4 in v3.items():
                                    if (isinstance(v4, str)):
                                        f.write("                {}='{}',\n".format(k4, v4))
                                    else:
                                        f.write("                {}={},\n".format(k4, v4))
                                f.write("            ),\n")
                        f.write("        ),\n")
                    elif (k2 == 'model'):
                        f.write("        model=dict(\n")
                        for k3, v3 in v2.items():
                            if (isinstance(v3, str)):
                                f.write("            {}='{}',\n".format(k3, v3))
                            else:
                                f.write("            {}={},\n".format(k3, v3))
                        f.write("        ),\n    ),\n)\n")
                    elif (k2 == 'other'):
                        f.write("        other=dict(\n")
                        for k3, v3 in v2.items():
                            if (k3 == 'replay_buffer'):
                                f.write("            replay_buffer=dict(\n")
                                for k4, v4 in v3.items():
                                    if (k4 != 'monitor'):
                                        if (isinstance(v4, str)):
                                            f.write("                {}='{}',\n".format(k4, v4))
                                        else:
                                            f.write("                {}={},\n".format(k4, v4))
                                    else:
                                        if (k4 == 'monitor'):
                                            f.write("                monitor=dict(\n")
                                            for k5, v5 in v4.items():
                                                if (k5 == 'log_path'):
                                                    if (isinstance(v5, str)):
                                                        f.write("                    {}='{}',\n".format(k5, v5))
                                                    else:
                                                        f.write("                    {}={},\n".format(k5, v5))
                                                else:
                                                    f.write("                    {}=dict(\n".format(k5))
                                                    for k6, v6 in v5.items():
                                                        if (isinstance(v6, str)):
                                                            f.write("                        {}='{}',\n".format(k6, v6))
                                                        else:
                                                            f.write("                        {}={},\n".format(k6, v6))
                                                    f.write("                    ),\n")
                                            f.write("                ),\n")
                                f.write("            ),\n")
                        f.write("        ),\n")
        f.write('main_config = EasyDict(main_config)\n')
        f.write('main_config = main_config\n')
        f.write('create_config = dict(\n')
        for k, v in config_.items():
            if (k == 'env'):
                f.write('    env=dict(\n')
                for k2, v2 in v.items():
                    if (k2 == 'type' or k2 == 'import_names'):
                        if isinstance(v2, str):
                            f.write("        {}='{}',\n".format(k2, v2))
                        else:
                            f.write("        {}={},\n".format(k2, v2))
                f.write("    ),\n")
                for k2, v2 in v.items():
                    if (k2 == 'manager'):
                        f.write('    env_manager=dict(\n')
                        for k3, v3 in v2.items():
                            if (v3 == 'cfg_type' or v3 == 'type'):
                                if (isinstance(v3, str)):
                                    f.write("        {}='{}',\n".format(k3, v3))
                                else:
                                    f.write("        {}={},\n".format(k3, v3))
                f.write("    ),\n")
        f.write("    policy=dict(type='{}'),\n".format(config_.policy.type[0:len(config_.policy.type) - 8]))
        f.write(")\n")
        f.write('create_config = EasyDict(create_config)\n')
        f.write('create_config = create_config\n')


def compile_buffer_config(policy_cfg: EasyDict, user_cfg: EasyDict, buffer_cls: 'IBuffer') -> EasyDict:  # noqa

    def _compile_buffer_config(policy_buffer_cfg, user_buffer_cfg, buffer_cls):

        if buffer_cls is None:
            assert 'type' in policy_buffer_cfg, "please indicate buffer type in create_cfg"
            buffer_cls = get_buffer_cls(policy_buffer_cfg)
        buffer_cfg = deep_merge_dicts(buffer_cls.default_config(), policy_buffer_cfg)
        buffer_cfg = deep_merge_dicts(buffer_cfg, user_buffer_cfg)
        return buffer_cfg

    policy_multi_buffer = policy_cfg.other.replay_buffer.get('multi_buffer', False)
    user_multi_buffer = user_cfg.policy.get('other', {}).get('replay_buffer', {}).get('multi_buffer', False)
    assert not user_multi_buffer or user_multi_buffer == policy_multi_buffer, "For multi_buffer, \
        user_cfg({}) and policy_cfg({}) must be in accordance".format(user_multi_buffer, policy_multi_buffer)
    multi_buffer = policy_multi_buffer
    if not multi_buffer:
        policy_buffer_cfg = policy_cfg.other.replay_buffer
        user_buffer_cfg = user_cfg.policy.get('other', {}).get('replay_buffer', {})
        return _compile_buffer_config(policy_buffer_cfg, user_buffer_cfg, buffer_cls)
    else:
        return_cfg = EasyDict()
        for buffer_name in policy_cfg.other.replay_buffer:  # Only traverse keys in policy_cfg
            if buffer_name == 'multi_buffer':
                continue
            policy_buffer_cfg = policy_cfg.other.replay_buffer[buffer_name]
            user_buffer_cfg = user_cfg.policy.get('other', {}).get('replay_buffer', {}).get('buffer_name', {})
            return_cfg[buffer_name] = _compile_buffer_config(
                policy_buffer_cfg, user_buffer_cfg, buffer_cls[buffer_name]
            )
            return_cfg[buffer_name].name = buffer_name
        return return_cfg


def compile_collector_config(
        policy_cfg: EasyDict,
        user_cfg: EasyDict,
        collector_cls: 'ISerialCollector'  # noqa
) -> EasyDict:
    policy_collector_cfg = policy_cfg.collect.collector
    user_collector_cfg = user_cfg.policy.get('collect', {}).get('collector', {})
    # step1: get collector class
    # two cases: create cfg merged in policy_cfg, collector class, and class has higher priority
    if collector_cls is None:
        assert 'type' in policy_collector_cfg, "please indicate collector type in create_cfg"
        # use type to get collector_cls
        collector_cls = get_serial_collector_cls(policy_collector_cfg)
    # step2: policy collector cfg merge to collector cfg
    collector_cfg = deep_merge_dicts(collector_cls.default_config(), policy_collector_cfg)
    # step3: user collector cfg merge to the step2 config
    collector_cfg = deep_merge_dicts(collector_cfg, user_collector_cfg)

    return collector_cfg


policy_config_template = dict(
    model=dict(),
    learn=dict(learner=dict()),
    collect=dict(collector=dict()),
    eval=dict(evaluator=dict()),
    other=dict(replay_buffer=dict()),
)
policy_config_template = EasyDict(policy_config_template)
env_config_template = dict(manager=dict(), )
env_config_template = EasyDict(env_config_template)


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
        # for compatibility
        if 'collector' not in create_cfg:
            create_cfg.collector = dict(type='sample')
        if 'replay_buffer' not in create_cfg:
            create_cfg.replay_buffer = dict(type='priority')
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
        env_config = deep_merge_dicts(env_config_template, env_config)
        env_config.update(create_cfg.env)
        env_config.manager = deep_merge_dicts(env_manager.default_config(), env_config.manager)
        env_config.manager.update(create_cfg.env_manager)
        policy_config = policy.default_config()
        policy_config = deep_merge_dicts(policy_config_template, policy_config)
        policy_config.update(create_cfg.policy)
        policy_config.collect.collector.update(create_cfg.collector)
        policy_config.other.replay_buffer.update(create_cfg.replay_buffer)

        policy_config.other.commander = BaseSerialCommander.default_config()
    else:
        if 'default_config' in dir(env):
            env_config = env.default_config()
        else:
            env_config = EasyDict()  # env does not have default_config
        env_config = deep_merge_dicts(env_config_template, env_config)
        env_config.manager = deep_merge_dicts(env_manager.default_config(), env_config.manager)
        policy_config = policy.default_config()
        policy_config = deep_merge_dicts(policy_config_template, policy_config)
    policy_config.learn.learner = deep_merge_dicts(
        learner.default_config(),
        policy_config.learn.learner,
    )
    policy_config.collect.collector = compile_collector_config(policy_config, cfg, collector)
    policy_config.eval.evaluator = deep_merge_dicts(
        evaluator.default_config(),
        policy_config.eval.evaluator,
    )
    policy_config.other.replay_buffer = compile_buffer_config(policy_config, cfg, buffer)
    default_config = EasyDict({'env': env_config, 'policy': policy_config})
    cfg = deep_merge_dicts(default_config, cfg)
    cfg.seed = seed
    # check important key in config
    assert all([k in cfg.env for k in ['n_evaluator_episode', 'stop_value']]), cfg.env
    cfg.policy.eval.evaluator.stop_value = cfg.env.stop_value
    cfg.policy.eval.evaluator.n_episode = cfg.env.n_evaluator_episode
    if save_cfg:
        save_config(cfg, save_path, save_formatted=True)
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
        coordinator_port: Optional[int] = None,
        learner_port: Optional[int] = None,
        collector_port: Optional[int] = None,
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
    elif platform == 'k8s':
        cfg = parallel_transform_k8s(
            EasyDict({
                'main': cfg,
                'system': system_cfg
            }),
            coordinator_port=coordinator_port,
            learner_port=learner_port,
            collector_port=collector_port
        )
    else:
        raise KeyError("not support platform type: {}".format(platform))
    cfg.seed = seed

    cfg.system.coordinator = deep_merge_dicts(Coordinator.default_config(), cfg.system.coordinator)
    if save_cfg:
        save_config(cfg, save_path)
    return cfg
