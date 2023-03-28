from dataclasses import dataclass
from typing import Optional, Union
from ditk import logging
from easydict import EasyDict
import os
import torch
import treetensor.torch as ttorch
import numpy as np
from ding.framework import task, OnlineRLContext
from ding.framework.middleware import CkptSaver, multistep_trainer, \
    wandb_online_logger, offline_data_saver, termination_checker, interaction_evaluator, StepCollector, data_pusher, \
    OffPolicyLearner, final_ctx_saver, nstep_reward_enhancer, eps_greedy_handler
from ding.envs import BaseEnv, BaseEnvManagerV2, SubprocessEnvManagerV2
from ding.policy import DQNPolicy
from ding.utils import set_pkg_seed
from ding.config import save_config_py, compile_config
from ding.model import DQN
from ding.model import model_wrap
from ding.data import DequeBuffer
from ding.bonus.config import get_instance_config, get_instance_env


@dataclass
class TrainingReturn:
    '''
    Attributions
    wandb_url: The weight & biases (wandb) project url of the trainning experiment.
    '''
    wandb_url: str


@dataclass
class EvalReturn:
    '''
    Attributions
    eval_value: The mean of evaluation return.
    eval_value_std: The standard deviation of evaluation return.
    '''
    eval_value: np.float32
    eval_value_std: np.float32


class DQNOffpolicyAgent:
    supported_env_list = [
        'lunarlander_discrete',
    ]
    algorithm = 'DQN'

    def __init__(
            self,
            env: Union[str, BaseEnv],
            seed: int = 0,
            exp_name: str = None,
            model: Optional[torch.nn.Module] = None,
            cfg: Optional[Union[EasyDict, dict, str]] = None,
            policy_state_dict: str = None,
    ) -> None:
        if isinstance(env, str):
            assert env in DQNOffpolicyAgent.supported_env_list, "Please use supported envs: {}".format(
                DQNOffpolicyAgent.supported_env_list
            )
            self.env = get_instance_env(env)
            if cfg is None:
                # 'It should be default env tuned config'
                cfg = get_instance_config(env, algorithm=DQNOffpolicyAgent.algorithm)
            else:
                assert isinstance(cfg, EasyDict), "Please use EasyDict as config data type."

            if exp_name is not None:
                cfg.exp_name = exp_name
            self.cfg = compile_config(cfg, policy=DQNPolicy)
            self.exp_name = self.cfg.exp_name

        elif isinstance(env, BaseEnv):
            self.cfg = compile_config(cfg, policy=DQNPolicy)
            raise NotImplementedError
        else:
            raise TypeError("not support env type: {}, only strings and instances of `BaseEnv` now".format(type(env)))
        logging.getLogger().setLevel(logging.INFO)
        self.seed = seed
        set_pkg_seed(self.seed, use_cuda=self.cfg.policy.cuda)
        if not os.path.exists(self.exp_name):
            os.makedirs(self.exp_name)
        save_config_py(self.cfg, os.path.join(self.exp_name, 'policy_config.py'))
        if model is None:
            model = DQN(**self.cfg.policy.model)
        self.buffer_ = DequeBuffer(size=self.cfg.policy.other.replay_buffer.replay_buffer_size)
        self.policy = DQNPolicy(self.cfg.policy, model=model)
        if policy_state_dict is not None:
            self.policy.learn_mode.load_state_dict(policy_state_dict)

    def train(
            self,
            step: int = int(1e7),
            collector_env_num: int = 4,
            evaluator_env_num: int = 4,
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

        with task.start(ctx=OnlineRLContext()):
            task.use(interaction_evaluator(self.cfg, self.policy.eval_mode, evaluator_env))
            task.use(eps_greedy_handler(self.cfg))
            # task.use(
            #     StepCollector(
            #         self.cfg,
            #         self.policy.collect_mode,
            #         collector_env,
            #         random_collect_size=self.cfg.policy.random_collect_size
            #     )
            # )
            task.use(StepCollector(self.cfg, self.policy.collect_mode, collector_env))
            task.use(nstep_reward_enhancer(self.cfg))
            task.use(data_pusher(self.cfg, self.buffer_))
            task.use(OffPolicyLearner(self.cfg, self.policy.learn_mode, self.buffer_))
            task.use(
                CkptSaver(
                    policy=self.policy,
                    save_dir=os.path.join(self.cfg["exp_name"], "model"),
                    train_freq=n_iter_save_ckpt
                )
            )
            task.use(
                wandb_online_logger(
                    metric_list=self.policy.monitor_vars(),
                    model=self.policy._model,
                    anonymous=True,
                    project_name=self.exp_name
                )
            )
            task.use(termination_checker(max_env_step=step))
            task.use(final_ctx_saver(name=self.cfg["exp_name"]))
            task.run()

        return TrainingReturn(wandb_url=task.ctx.wandb_url)

    def deploy(self, enable_save_replay: bool = False, replay_save_path: str = None, debug: bool = False) -> None:
        if debug:
            logging.getLogger().setLevel(logging.DEBUG)
        # define env and policy
        env = self.env.clone()
        env.seed(self.seed, dynamic_seed=False)

        if enable_save_replay and replay_save_path:
            env.enable_save_replay(replay_path=replay_save_path)
        elif enable_save_replay:
            env.enable_save_replay(replay_path=os.path.join(self.exp_name, 'videos'))
        else:
            logging.warning('No video would be generated during the deploy.')

        def single_env_forward_wrapper(forward_fn, cuda=True):

            forward_fn = model_wrap(forward_fn, wrapper_name='argmax_sample').forward

            def _forward(obs):
                # unsqueeze means add batch dim, i.e. (O, ) -> (1, O)
                obs = ttorch.as_tensor(obs).unsqueeze(0)
                if cuda and torch.cuda.is_available():
                    obs = obs.cuda()
                action = forward_fn(obs)["action"]
                # squeeze means delete batch dim, i.e. (1, A) -> (A, )
                action = action.squeeze(0).detach().cpu().numpy()
                return action

            return _forward

        forward_fn = single_env_forward_wrapper(self.policy._model, self.cfg.policy.cuda)

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
        logging.info(f'DQN deploy is finished, final episode return with {step} steps is: {return_}')

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
        env = self._setup_env_manager(env_num, context, debug)

        if save_data_path is None:
            save_data_path = os.path.join(self.exp_name, 'demo_data')

        # main execution task
        with task.start(ctx=OnlineRLContext()):
            task.use(
                StepCollector(
                    self.cfg, self.policy.collect_mode, env, random_collect_size=self.cfg.policy.random_collect_size
                )
            )
            task.use(offline_data_saver(save_data_path, data_type='hdf5'))
            task.run(max_step=1)
        logging.info(
            f'DQN collecting is finished, more than {n_sample} samples are collected and saved in `{save_data_path}`'
        )

    def batch_evaluate(
            self,
            env_num: int = 4,
            n_evaluator_episode: int = 4,
            context: Optional[str] = None,
            debug: bool = False
    ) -> EvalReturn:
        if debug:
            logging.getLogger().setLevel(logging.DEBUG)
        # define env and policy
        env = self._setup_env_manager(env_num, context, debug)

        evaluate_cfg = self.cfg
        evaluate_cfg.env.n_evaluator_episode = n_evaluator_episode

        # main execution task
        with task.start(ctx=OnlineRLContext()):
            task.use(interaction_evaluator(self.cfg, self.policy.eval_mode, env))
            task.run(max_step=1)

        return EvalReturn(eval_value=task.ctx.eval_value, eval_value_std=task.ctx.eval_value_std)

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
