from typing import Optional, Union
from ditk import logging
from easydict import EasyDict
import os
import gym
import torch
from ding.framework import task, OnlineRLContext
from ding.framework.middleware import interaction_evaluator_ttorch, PPOFStepCollector, multistep_trainer, CkptSaver, \
    wandb_online_logger, offline_data_saver, termination_checker, ppof_adv_estimator
from ding.envs import BaseEnv, BaseEnvManagerV2, SubprocessEnvManagerV2
from ding.policy import PPOFPolicy, single_env_forward_wrapper_ttorch
from ding.utils import set_pkg_seed
from ding.config import save_config_py
from .model import PPOFModel
from .config import get_instance_config, get_instance_env, get_hybrid_shape

class TrainingReturn:
    wandb_url:str

class PPOF:
    supported_env_list = [
        # common
        'lunarlander_discrete',
        'lunarlander_continuous',
        'bipedalwalker',
        # ch2: action
        'rocket_landing',
        'drone_fly',
        'hybrid_moving',
        # ch3: obs
        'evogym_carrier',
        'mario',
        'di_sheep',
        'procgen_bigfish',
    ]

    def __init__(
            self,
            env: Union[str, BaseEnv],
            seed: int = 0,
            exp_name: str = 'default_experiment',
            model: Optional[torch.nn.Module] = None,
            cfg: Optional[EasyDict] = None
    ) -> None:
        if isinstance(env, str):
            assert env in PPOF.supported_env_list, "Please use supported envs: {}".format(PPOF.supported_env_list)
            self.env = get_instance_env(env)
            if cfg is None:
                # 'It should be default env tuned config'
                self.cfg = get_instance_config(env)
            else:
                self.cfg = cfg
        elif isinstance(env, BaseEnv):
            self.cfg = cfg
            raise NotImplementedError
        else:
            raise TypeError("not support env type: {}, only strings and instances of `BaseEnv` now".format(type(env)))
        logging.getLogger().setLevel(logging.INFO)
        self.seed = seed
        set_pkg_seed(self.seed)
        self.exp_name = exp_name
        if not os.path.exists(self.exp_name):
            os.makedirs(self.exp_name)
        save_config_py(self.cfg, os.path.join(self.exp_name, 'policy_config.py'))

        action_space = self.env.action_space
        if isinstance(action_space, gym.spaces.Discrete):
            action_shape = action_space.n
        elif isinstance(action_space, gym.spaces.Tuple):
            action_shape = get_hybrid_shape(action_space)
        else:
            action_shape = action_space.shape
        if model is None:
            model = PPOFModel(
                self.env.observation_space.shape, action_shape, action_space=self.cfg.action_space, **self.cfg.model
            )
        self.policy = PPOFPolicy(self.cfg, model=model)

    def load_policy(self,policy_state_dict, config):
        self.policy.load_state_dict(policy_state_dict)
        self.policy._cfg = config


    def train(
            self,
            step: int = int(1e7),
            collector_env_num: int = 4,
            evaluator_env_num: int = 4,
            n_iter_log_show: int = 500,
            n_iter_save_ckpt: int = 1000,
            context: Optional[str] = None,
            debug: bool = False
    ) -> TrainingReturn:
        if debug:
            logging.getLogger().setLevel(logging.DEBUG)
        logging.debug(self.policy._model)
        # define env and policy
        collector_env = self._setup_env_manager(collector_env_num, context, debug)
        evaluator_env = self._setup_env_manager(evaluator_env_num, context, debug)
        wandb_url_return=[]

        with task.start(ctx=OnlineRLContext()):
            task.use(interaction_evaluator_ttorch(self.seed, self.policy, evaluator_env))
            task.use(PPOFStepCollector(self.seed, self.policy, collector_env, self.cfg.n_sample))
            task.use(ppof_adv_estimator(self.policy))
            task.use(multistep_trainer(self.policy, log_freq=n_iter_log_show))
            task.use(CkptSaver(self.policy, save_dir=self.exp_name, train_freq=n_iter_save_ckpt))
            task.use(wandb_online_logger(self.exp_name, metric_list=self.policy.monitor_vars(), anonymous=True, project_name=self.exp_name, wandb_url_return=wandb_url_return))
            task.use(termination_checker(max_env_step=step))
            task.run()
        
        return TrainingReturn(wandb_url=wandb_url_return[0])

    def deploy(self, ckpt_path: str = None, enable_save_replay: bool = False, debug: bool = False) -> None:
        if debug:
            logging.getLogger().setLevel(logging.DEBUG)
        # define env and policy
        env = self.env.clone()
        env.seed(self.seed, dynamic_seed=False)
        if enable_save_replay:
            env.enable_save_replay(replay_path=os.path.join(self.exp_name, 'videos'))
        if ckpt_path is None:
            ckpt_path = os.path.join(self.exp_name, 'ckpt/eval.pth.tar')
        state_dict = torch.load(ckpt_path, map_location='cpu')
        self.policy.load_state_dict(state_dict)
        forward_fn = single_env_forward_wrapper_ttorch(self.policy.eval)

        # main loop
        return_ = 0.
        step = 0
        obs = env.reset()
        while True:
            action = forward_fn(obs)
            obs, rew, done, info = env.step(action)
            return_ += rew
            step += 1
            if done:
                break
        logging.info(f'PPOF deploy is finished, final episode return with {step} steps is: {return_}')

    def collect_data(
            self,
            env_num: int = 8,
            ckpt_path: Optional[str] = None,
            save_data_path: Optional[str] = None,
            n_sample: Optional[int] = None,
            n_episode: Optional[int] = None,
            context: Optional[str] = None,
            debug: bool = False
    ) -> None:
        if debug:
            logging.getLogger().setLevel(logging.DEBUG)
        if n_episode is not None:
            raise NotImplementedError
        # define env and policy
        env = self._setup_env_manager(env_num, context, debug)
        if ckpt_path is None:
            ckpt_path = os.path.join(self.exp_name, 'ckpt/eval.pth.tar')
        if save_data_path is None:
            save_data_path = os.path.join(self.exp_name, 'demo_data')
        state_dict = torch.load(ckpt_path, map_location='cpu')
        self.policy.load_state_dict(state_dict)

        # main execution task
        with task.start(ctx=OnlineRLContext()):
            task.use(PPOFStepCollector(self.seed, self.policy, env, n_sample))
            task.use(offline_data_saver(save_data_path, data_type='hdf5'))
            task.run(max_step=1)
        logging.info(
            f'PPOF collecting is finished, more than {n_sample} samples are collected and saved in `{save_data_path}`'
        )

    def batch_evaluate(
            self,
            env_num: int = 4,
            n_evaluator_episode: int = 4,
            context: Optional[str] = None,
            debug: bool = False,
            render: bool = False,
            replay_video_path: str = None,
    ) -> None:
        if debug:
            logging.getLogger().setLevel(logging.DEBUG)
        # define env and policy
        env = self._setup_env_manager(env_num, context, debug)

        # main execution task
        with task.start(ctx=OnlineRLContext()):
            task.use(interaction_evaluator_ttorch(self.seed, self.policy, env, n_evaluator_episode, render=render, replay_video_path=replay_video_path))
            task.run(max_step=1)

    def _setup_env_manager(self, env_num: int, context: Optional[str] = None, debug: bool = False) -> BaseEnvManagerV2:
        if debug:
            env_cls = BaseEnvManagerV2
            manager_cfg = env_cls.default_config()
        else:
            env_cls = SubprocessEnvManagerV2
            manager_cfg = env_cls.default_config()
            if context is not None:
                manager_cfg.context = context
        return env_cls([self.env.clone for _ in range(env_num)], manager_cfg)
