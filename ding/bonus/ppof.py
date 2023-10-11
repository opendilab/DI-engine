from typing import Optional, Union, List
from ditk import logging
from easydict import EasyDict
from functools import partial
import os
import gym
import gymnasium
import numpy as np
import torch
from ding.framework import task, OnlineRLContext
from ding.framework.middleware import interaction_evaluator_ttorch, PPOFStepCollector, multistep_trainer, CkptSaver, \
    wandb_online_logger, offline_data_saver, termination_checker, ppof_adv_estimator
from ding.envs import BaseEnv, BaseEnvManagerV2, SubprocessEnvManagerV2
from ding.policy import PPOFPolicy, single_env_forward_wrapper_ttorch
from ding.utils import set_pkg_seed
from ding.utils import get_env_fps, render
from ding.config import save_config_py
from .model import PPOFModel
from .config import get_instance_config, get_instance_env, get_hybrid_shape
from ding.bonus.common import TrainingReturn, EvalReturn


class PPOF:
    supported_env_list = [
        # common
        'LunarLander-v2',
        'LunarLanderContinuous-v2',
        'BipedalWalker-v3',
        'acrobot',
        # ch2: action
        'rocket_landing',
        'drone_fly',
        'hybrid_moving',
        # ch3: obs
        'evogym_carrier',
        'mario',
        'di_sheep',
        'procgen_bigfish',
        # ch4: reward
        'minigrid_fourroom',
        'metadrive',
        # atari
        'BowlingNoFrameskip-v4',
        'BreakoutNoFrameskip-v4',
        'GopherNoFrameskip-v4'
        'KangarooNoFrameskip-v4',
        'PongNoFrameskip-v4',
        'QbertNoFrameskip-v4',
        'SpaceInvadersNoFrameskip-v4',
        # mujoco
        'Hopper-v3',
        'HalfCheetah-v3',
        'Walker2d-v3',
    ]

    def __init__(
            self,
            env_id: str = None,
            env: BaseEnv = None,
            seed: int = 0,
            exp_name: str = None,
            model: Optional[torch.nn.Module] = None,
            cfg: Optional[Union[EasyDict, dict]] = None,
            policy_state_dict: str = None
    ) -> None:
        assert env_id is not None or cfg is not None, "Please specify env_id or cfg."

        if cfg is not None and not isinstance(cfg, EasyDict):
            cfg = EasyDict(cfg)

        if env_id is not None:
            assert env_id in PPOF.supported_env_list, "Please use supported envs: {}".format(PPOF.supported_env_list)
            if cfg is None:
                cfg = get_instance_config(env_id, algorithm="PPOF")

            if not hasattr(cfg, "env_id"):
                cfg.env_id = env_id
            assert cfg.env_id == env_id, "env_id in cfg should be the same as env_id in args."
        else:
            assert hasattr(cfg, "env_id"), "Please specify env_id in cfg."
            assert cfg.env_id in PPOF.supported_env_list, "Please use supported envs: {}".format(
                PPOF.supported_env_list
            )

        if exp_name is not None:
            cfg.exp_name = exp_name
        elif not hasattr(cfg, "exp_name"):
            cfg.exp_name = "{}-{}".format(cfg.env_id, "PPO")
        self.cfg = cfg
        self.exp_name = self.cfg.exp_name

        if env is None:
            self.env = get_instance_env(self.cfg.env_id)
        else:
            self.env = env

        logging.getLogger().setLevel(logging.INFO)
        self.seed = seed
        set_pkg_seed(self.seed, use_cuda=self.cfg.cuda)

        if not os.path.exists(self.exp_name):
            os.makedirs(self.exp_name)
        save_config_py(self.cfg, os.path.join(self.exp_name, 'policy_config.py'))

        action_space = self.env.action_space
        if isinstance(action_space, (gym.spaces.Discrete, gymnasium.spaces.Discrete)):
            action_shape = int(action_space.n)
        elif isinstance(action_space, (gym.spaces.Tuple, gymnasium.spaces.Tuple)):
            action_shape = get_hybrid_shape(action_space)
        else:
            action_shape = action_space.shape

        # Three types of value normalization is supported currently
        assert self.cfg.value_norm in ['popart', 'value_rescale', 'symlog', 'baseline']
        if model is None:
            if self.cfg.value_norm != 'popart':
                model = PPOFModel(
                    self.env.observation_space.shape,
                    action_shape,
                    action_space=self.cfg.action_space,
                    **self.cfg.model
                )
            else:
                model = PPOFModel(
                    self.env.observation_space.shape,
                    action_shape,
                    action_space=self.cfg.action_space,
                    popart_head=True,
                    **self.cfg.model
                )
        self.policy = PPOFPolicy(self.cfg, model=model)
        if policy_state_dict is not None:
            self.policy.load_state_dict(policy_state_dict)
        self.checkpoint_save_dir = os.path.join(self.exp_name, "ckpt")

    def train(
        self,
        step: int = int(1e7),
        collector_env_num: int = 4,
        evaluator_env_num: int = 4,
        n_iter_log_show: int = 500,
        n_iter_save_ckpt: int = 1000,
        context: Optional[str] = None,
        reward_model: Optional[str] = None,
        debug: bool = False,
        wandb_sweep: bool = False,
    ) -> TrainingReturn:
        if debug:
            logging.getLogger().setLevel(logging.DEBUG)
        logging.debug(self.policy._model)
        # define env and policy
        collector_env = self._setup_env_manager(collector_env_num, context, debug, 'collector')
        evaluator_env = self._setup_env_manager(evaluator_env_num, context, debug, 'evaluator')

        if reward_model is not None:
            # self.reward_model = create_reward_model(reward_model, self.cfg.reward_model)
            pass

        with task.start(ctx=OnlineRLContext()):
            task.use(interaction_evaluator_ttorch(self.seed, self.policy, evaluator_env))
            task.use(CkptSaver(self.policy, save_dir=self.checkpoint_save_dir, train_freq=n_iter_save_ckpt))
            task.use(PPOFStepCollector(self.seed, self.policy, collector_env, self.cfg.n_sample))
            task.use(ppof_adv_estimator(self.policy))
            task.use(multistep_trainer(self.policy, log_freq=n_iter_log_show))
            task.use(
                wandb_online_logger(
                    metric_list=self.policy.monitor_vars(),
                    model=self.policy._model,
                    anonymous=True,
                    project_name=self.exp_name,
                    wandb_sweep=wandb_sweep,
                )
            )
            task.use(termination_checker(max_env_step=step))
            task.run()

        return TrainingReturn(wandb_url=task.ctx.wandb_url)

    def deploy(
            self,
            enable_save_replay: bool = False,
            concatenate_all_replay: bool = False,
            replay_save_path: str = None,
            seed: Optional[Union[int, List]] = None,
            debug: bool = False
    ) -> EvalReturn:
        if debug:
            logging.getLogger().setLevel(logging.DEBUG)
        # define env and policy
        env = self.env.clone(caller='evaluator')

        if seed is not None and isinstance(seed, int):
            seeds = [seed]
        elif seed is not None and isinstance(seed, list):
            seeds = seed
        else:
            seeds = [self.seed]

        returns = []
        images = []
        if enable_save_replay:
            replay_save_path = os.path.join(self.exp_name, 'videos') if replay_save_path is None else replay_save_path
            env.enable_save_replay(replay_path=replay_save_path)
        else:
            logging.warning('No video would be generated during the deploy.')
            if concatenate_all_replay:
                logging.warning('concatenate_all_replay is set to False because enable_save_replay is False.')
                concatenate_all_replay = False

        forward_fn = single_env_forward_wrapper_ttorch(self.policy.eval, self.cfg.cuda)

        # reset first to make sure the env is in the initial state
        # env will be reset again in the main loop
        env.reset()

        for seed in seeds:
            env.seed(seed, dynamic_seed=False)
            return_ = 0.
            step = 0
            obs = env.reset()
            images.append(render(env)[None]) if concatenate_all_replay else None
            while True:
                action = forward_fn(obs)
                obs, rew, done, info = env.step(action)
                images.append(render(env)[None]) if concatenate_all_replay else None
                return_ += rew
                step += 1
                if done:
                    break
            logging.info(f'DQN deploy is finished, final episode return with {step} steps is: {return_}')
            returns.append(return_)

        env.close()

        if concatenate_all_replay:
            images = np.concatenate(images, axis=0)
            import imageio
            imageio.mimwrite(os.path.join(replay_save_path, 'deploy.mp4'), images, fps=get_env_fps(env))

        return EvalReturn(eval_value=np.mean(returns), eval_value_std=np.std(returns))

    def collect_data(
            self,
            env_num: int = 8,
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
        env = self._setup_env_manager(env_num, context, debug, 'collector')
        if save_data_path is None:
            save_data_path = os.path.join(self.exp_name, 'demo_data')

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
    ) -> EvalReturn:
        if debug:
            logging.getLogger().setLevel(logging.DEBUG)
        # define env and policy
        env = self._setup_env_manager(env_num, context, debug, 'evaluator')

        # reset first to make sure the env is in the initial state
        # env will be reset again in the main loop
        env.launch()
        env.reset()

        # main execution task
        with task.start(ctx=OnlineRLContext()):
            task.use(interaction_evaluator_ttorch(
                self.seed,
                self.policy,
                env,
                n_evaluator_episode,
            ))
            task.run(max_step=1)

        return EvalReturn(eval_value=task.ctx.eval_value, eval_value_std=task.ctx.eval_value_std)

    def _setup_env_manager(
            self,
            env_num: int,
            context: Optional[str] = None,
            debug: bool = False,
            caller: str = 'collector'
    ) -> BaseEnvManagerV2:
        assert caller in ['evaluator', 'collector']
        if debug:
            env_cls = BaseEnvManagerV2
            manager_cfg = env_cls.default_config()
        else:
            env_cls = SubprocessEnvManagerV2
            manager_cfg = env_cls.default_config()
            if context is not None:
                manager_cfg.context = context
        return env_cls([partial(self.env.clone, caller) for _ in range(env_num)], manager_cfg)

    @property
    def best(self):
        best_model_file_path = os.path.join(self.checkpoint_save_dir, "eval.pth.tar")
        # Load best model if it exists
        if os.path.exists(best_model_file_path):
            policy_state_dict = torch.load(best_model_file_path, map_location=torch.device("cpu"))
            self.policy.learn_mode.load_state_dict(policy_state_dict)
        return self
