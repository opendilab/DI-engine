from typing import Optional, Union, List
from ditk import logging
from easydict import EasyDict
import os
import numpy as np
import torch
import treetensor.torch as ttorch
from ding.framework import task, OnlineRLContext
from ding.framework.middleware import CkptSaver, \
    wandb_online_logger, offline_data_saver, termination_checker, interaction_evaluator, StepCollector, data_pusher, \
    OffPolicyLearner, final_ctx_saver, eps_greedy_handler, nstep_reward_enhancer
from ding.envs import BaseEnv
from ding.envs import setup_ding_env_manager
from ding.policy import C51Policy
from ding.utils import set_pkg_seed
from ding.utils import get_env_fps, render
from ding.config import save_config_py, compile_config
from ding.model import C51DQN
from ding.model import model_wrap
from ding.data import DequeBuffer
from ding.bonus.common import TrainingReturn, EvalReturn
from ding.config.example.C51 import supported_env_cfg
from ding.config.example.C51 import supported_env


class C51Agent:
    supported_env_list = list(supported_env_cfg.keys())

    def __init__(
            self,
            env_id: str = None,
            env: BaseEnv = None,
            seed: int = 0,
            exp_name: str = None,
            model: Optional[torch.nn.Module] = None,
            cfg: Optional[Union[EasyDict, dict]] = None,
            policy_state_dict: str = None,
    ) -> None:
        assert env_id is not None or cfg is not None, "Please specify env_id or cfg."

        if cfg is not None and not isinstance(cfg, EasyDict):
            cfg = EasyDict(cfg)

        if env_id is not None:
            assert env_id in C51Agent.supported_env_list, "Please use supported envs: {}".format(
                C51Agent.supported_env_list
            )
            if cfg is None:
                cfg = supported_env_cfg[env_id]
            else:
                assert cfg.env.env_id == env_id, "env_id in cfg should be the same as env_id in args."
        else:
            assert hasattr(cfg.env, "env_id"), "Please specify env_id in cfg."
            assert cfg.env.env_id in C51Agent.supported_env_list, "Please use supported envs: {}".format(
                C51Agent.supported_env_list
            )
        default_policy_config = EasyDict({"policy": C51Policy.default_config()})
        default_policy_config.update(cfg)
        cfg = default_policy_config

        if exp_name is not None:
            cfg.exp_name = exp_name
        self.cfg = compile_config(cfg, policy=C51Policy)
        self.exp_name = self.cfg.exp_name
        if env is None:
            self.env = supported_env[cfg.env.env_id](cfg=cfg.env)
        else:
            assert isinstance(env, BaseEnv), "Please use BaseEnv as env data type."
            self.env = env

        logging.getLogger().setLevel(logging.INFO)
        self.seed = seed
        set_pkg_seed(self.seed, use_cuda=self.cfg.policy.cuda)
        if not os.path.exists(self.exp_name):
            os.makedirs(self.exp_name)
        save_config_py(self.cfg, os.path.join(self.exp_name, 'policy_config.py'))
        if model is None:
            model = C51DQN(**self.cfg.policy.model)
        self.buffer_ = DequeBuffer(size=self.cfg.policy.other.replay_buffer.replay_buffer_size)
        self.policy = C51Policy(self.cfg.policy, model=model)
        if policy_state_dict is not None:
            self.policy.learn_mode.load_state_dict(policy_state_dict)
        self.checkpoint_save_dir = os.path.join(self.exp_name, "ckpt")

    def train(
        self,
        step: int = int(1e7),
        collector_env_num: int = None,
        evaluator_env_num: int = None,
        n_iter_save_ckpt: int = 1000,
        context: Optional[str] = None,
        debug: bool = False,
        wandb_sweep: bool = False,
    ) -> TrainingReturn:
        if debug:
            logging.getLogger().setLevel(logging.DEBUG)
        logging.debug(self.policy._model)
        # define env and policy
        collector_env_num = collector_env_num if collector_env_num else self.cfg.env.collector_env_num
        evaluator_env_num = evaluator_env_num if evaluator_env_num else self.cfg.env.evaluator_env_num
        collector_env = setup_ding_env_manager(self.env, collector_env_num, context, debug, 'collector')
        evaluator_env = setup_ding_env_manager(self.env, evaluator_env_num, context, debug, 'evaluator')

        with task.start(ctx=OnlineRLContext()):
            task.use(
                interaction_evaluator(
                    self.cfg,
                    self.policy.eval_mode,
                    evaluator_env,
                    render=self.cfg.policy.eval.render if hasattr(self.cfg.policy.eval, "render") else False
                )
            )
            task.use(CkptSaver(policy=self.policy, save_dir=self.checkpoint_save_dir, train_freq=n_iter_save_ckpt))
            task.use(eps_greedy_handler(self.cfg))
            task.use(
                StepCollector(
                    self.cfg,
                    self.policy.collect_mode,
                    collector_env,
                    random_collect_size=self.cfg.policy.random_collect_size
                    if hasattr(self.cfg.policy, 'random_collect_size') else 0,
                )
            )
            task.use(nstep_reward_enhancer(self.cfg))
            task.use(data_pusher(self.cfg, self.buffer_))
            task.use(OffPolicyLearner(self.cfg, self.policy.learn_mode, self.buffer_))
            task.use(
                wandb_online_logger(
                    metric_list=self.policy._monitor_vars_learn(),
                    model=self.policy._model,
                    anonymous=True,
                    project_name=self.exp_name,
                    wandb_sweep=wandb_sweep,
                )
            )
            task.use(termination_checker(max_env_step=step))
            task.use(final_ctx_saver(name=self.exp_name))
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
        env_num = env_num if env_num else self.cfg.env.collector_env_num
        env = setup_ding_env_manager(self.env, env_num, context, debug, 'collector')

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
            f'C51 collecting is finished, more than {n_sample} samples are collected and saved in `{save_data_path}`'
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
        env_num = env_num if env_num else self.cfg.env.evaluator_env_num
        env = setup_ding_env_manager(self.env, env_num, context, debug, 'evaluator')

        # reset first to make sure the env is in the initial state
        # env will be reset again in the main loop
        env.launch()
        env.reset()

        evaluate_cfg = self.cfg
        evaluate_cfg.env.n_evaluator_episode = n_evaluator_episode

        # main execution task
        with task.start(ctx=OnlineRLContext()):
            task.use(interaction_evaluator(self.cfg, self.policy.eval_mode, env))
            task.run(max_step=1)

        return EvalReturn(eval_value=task.ctx.eval_value, eval_value_std=task.ctx.eval_value_std)

    @property
    def best(self):
        best_model_file_path = os.path.join(self.checkpoint_save_dir, "eval.pth.tar")
        # Load best model if it exists
        if os.path.exists(best_model_file_path):
            policy_state_dict = torch.load(best_model_file_path, map_location=torch.device("cpu"))
            self.policy.learn_mode.load_state_dict(policy_state_dict)
        return self
