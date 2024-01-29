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
    OffPolicyLearner, final_ctx_saver, nstep_reward_enhancer, eps_greedy_handler
from ding.envs import BaseEnv
from ding.envs import setup_ding_env_manager
from ding.policy import SQLPolicy
from ding.utils import set_pkg_seed
from ding.utils import get_env_fps, render
from ding.config import save_config_py, compile_config
from ding.model import DQN
from ding.model import model_wrap
from ding.data import DequeBuffer
from ding.bonus.common import TrainingReturn, EvalReturn
from ding.config.example.SQL import supported_env_cfg
from ding.config.example.SQL import supported_env


class SQLAgent:
    """
    Overview:
        Class of agent for training, evaluation and deployment of Reinforcement learning algorithm \
        Soft Q-Learning(SQL).
        For more information about the system design of RL agent, please refer to \
        <https://di-engine-docs.readthedocs.io/en/latest/03_system/agent.html>.
    Interface:
        ``__init__``, ``train``, ``deploy``, ``collect_data``, ``batch_evaluate``, ``best``
    """
    supported_env_list = list(supported_env_cfg.keys())
    """
    Overview:
        List of supported envs.
    Examples:
        >>> from ding.bonus.sql import SQLAgent
        >>> print(SQLAgent.supported_env_list)
    """

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
        """
        Overview:
            Initialize agent for SQL algorithm.
        Arguments:
            - env_id (:obj:`str`): The environment id, which is a registered environment name in gym or gymnasium. \
                If ``env_id`` is not specified, ``env_id`` in ``cfg.env`` must be specified. \
                If ``env_id`` is specified, ``env_id`` in ``cfg.env`` will be ignored. \
                ``env_id`` should be one of the supported envs, which can be found in ``supported_env_list``.
            - env (:obj:`BaseEnv`): The environment instance for training and evaluation. \
                If ``env`` is not specified, `env_id`` or ``cfg.env.env_id`` must be specified. \
                ``env_id`` or ``cfg.env.env_id`` will be used to create environment instance. \
                If ``env`` is specified, ``env_id`` and ``cfg.env.env_id`` will be ignored.
            - seed (:obj:`int`): The random seed, which is set before running the program. \
                Default to 0.
            - exp_name (:obj:`str`): The name of this experiment, which will be used to create the folder to save \
                log data. Default to None. If not specified, the folder name will be ``env_id``-``algorithm``.
            - model (:obj:`torch.nn.Module`): The model of SQL algorithm, which should be an instance of class \
                :class:`ding.model.DQN`. \
                If not specified, a default model will be generated according to the configuration.
            - cfg (:obj:Union[EasyDict, dict]): The configuration of SQL algorithm, which is a dict. \
                Default to None. If not specified, the default configuration will be used. \
                The default configuration can be found in ``ding/config/example/SQL/gym_lunarlander_v2.py``.
            - policy_state_dict (:obj:`str`): The path of policy state dict saved by PyTorch a in local file. \
                If specified, the policy will be loaded from this file. Default to None.

        .. note::
            An RL Agent Instance can be initialized in two basic ways. \
            For example, we have an environment with id ``LunarLander-v2`` registered in gym, \
            and we want to train an agent with SQL algorithm with default configuration. \
            Then we can initialize the agent in the following ways:
                >>> agent = SQLAgent(env_id='LunarLander-v2')
            or, if we want can specify the env_id in the configuration:
                >>> cfg = {'env': {'env_id': 'LunarLander-v2'}, 'policy': ...... }
                >>> agent = SQLAgent(cfg=cfg)
            There are also other arguments to specify the agent when initializing.
            For example, if we want to specify the environment instance:
                >>> env = CustomizedEnv('LunarLander-v2')
                >>> agent = SQLAgent(cfg=cfg, env=env)
            or, if we want to specify the model:
                >>> model = DQN(**cfg.policy.model)
                >>> agent = SQLAgent(cfg=cfg, model=model)
            or, if we want to reload the policy from a saved policy state dict:
                >>> agent = SQLAgent(cfg=cfg, policy_state_dict='LunarLander-v2.pth.tar')
            Make sure that the configuration is consistent with the saved policy state dict.
        """

        assert env_id is not None or cfg is not None, "Please specify env_id or cfg."

        if cfg is not None and not isinstance(cfg, EasyDict):
            cfg = EasyDict(cfg)

        if env_id is not None:
            assert env_id in SQLAgent.supported_env_list, "Please use supported envs: {}".format(
                SQLAgent.supported_env_list
            )
            if cfg is None:
                cfg = supported_env_cfg[env_id]
            else:
                assert cfg.env.env_id == env_id, "env_id in cfg should be the same as env_id in args."
        else:
            assert hasattr(cfg.env, "env_id"), "Please specify env_id in cfg."
            assert cfg.env.env_id in SQLAgent.supported_env_list, "Please use supported envs: {}".format(
                SQLAgent.supported_env_list
            )
        default_policy_config = EasyDict({"policy": SQLPolicy.default_config()})
        default_policy_config.update(cfg)
        cfg = default_policy_config

        if exp_name is not None:
            cfg.exp_name = exp_name
        self.cfg = compile_config(cfg, policy=SQLPolicy)
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
            model = DQN(**self.cfg.policy.model)
        self.buffer_ = DequeBuffer(size=self.cfg.policy.other.replay_buffer.replay_buffer_size)
        self.policy = SQLPolicy(self.cfg.policy, model=model)
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
        """
        Overview:
            Train the agent with SQL algorithm for ``step`` iterations with ``collector_env_num`` collector \
            environments and ``evaluator_env_num`` evaluator environments. Information during training will be \
            recorded and saved by wandb.
        Arguments:
            - step (:obj:`int`): The total training environment steps of all collector environments. Default to 1e7.
            - collector_env_num (:obj:`int`): The collector environment number. Default to None. \
                If not specified, it will be set according to the configuration.
            - evaluator_env_num (:obj:`int`): The evaluator environment number. Default to None. \
                If not specified, it will be set according to the configuration.
            - n_iter_save_ckpt (:obj:`int`): The frequency of saving checkpoint every training iteration. \
                Default to 1000.
            - context (:obj:`str`): The multi-process context of the environment manager. Default to None. \
                It can be specified as ``spawn``, ``fork`` or ``forkserver``.
            - debug (:obj:`bool`): Whether to use debug mode in the environment manager. Default to False. \
                If set True, base environment manager will be used for easy debugging. Otherwise, \
                subprocess environment manager will be used.
            - wandb_sweep (:obj:`bool`): Whether to use wandb sweep, \
                which is a hyper-parameter optimization process for seeking the best configurations. \
                Default to False. If True, the wandb sweep id will be used as the experiment name.
        Returns:
            - (:obj:`TrainingReturn`): The training result, of which the attributions are:
                - wandb_url (:obj:`str`): The weight & biases (wandb) project url of the trainning experiment.
        """

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
            if "nstep" in self.cfg.policy and self.cfg.policy.nstep > 1:
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
        """
        Overview:
            Deploy the agent with SQL algorithm by interacting with the environment, during which the replay video \
            can be saved if ``enable_save_replay`` is True. The evaluation result will be returned.
        Arguments:
            - enable_save_replay (:obj:`bool`): Whether to save the replay video. Default to False.
            - concatenate_all_replay (:obj:`bool`): Whether to concatenate all replay videos into one video. \
                Default to False. If ``enable_save_replay`` is False, this argument will be ignored. \
                If ``enable_save_replay`` is True and ``concatenate_all_replay`` is False, \
                the replay video of each episode will be saved separately.
            - replay_save_path (:obj:`str`): The path to save the replay video. Default to None. \
                If not specified, the video will be saved in ``exp_name/videos``.
            - seed (:obj:`Union[int, List]`): The random seed, which is set before running the program. \
                Default to None. If not specified, ``self.seed`` will be used. \
                If ``seed`` is an integer, the agent will be deployed once. \
                If ``seed`` is a list of integers, the agent will be deployed once for each seed in the list.
            - debug (:obj:`bool`): Whether to use debug mode in the environment manager. Default to False. \
                If set True, base environment manager will be used for easy debugging. Otherwise, \
                subprocess environment manager will be used.
        Returns:
            - (:obj:`EvalReturn`): The evaluation result, of which the attributions are:
                - eval_value (:obj:`np.float32`): The mean of evaluation return.
                - eval_value_std (:obj:`np.float32`): The standard deviation of evaluation return.
        """

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
            logging.info(f'SQL deploy is finished, final episode return with {step} steps is: {return_}')
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
        """
        Overview:
            Collect data with SQL algorithm for ``n_episode`` episodes with ``env_num`` collector environments. \
            The collected data will be saved in ``save_data_path`` if specified, otherwise it will be saved in \
            ``exp_name/demo_data``.
        Arguments:
            - env_num (:obj:`int`): The number of collector environments. Default to 8.
            - save_data_path (:obj:`str`): The path to save the collected data. Default to None. \
                If not specified, the data will be saved in ``exp_name/demo_data``.
            - n_sample (:obj:`int`): The number of samples to collect. Default to None. \
                If not specified, ``n_episode`` must be specified.
            - n_episode (:obj:`int`): The number of episodes to collect. Default to None. \
                If not specified, ``n_sample`` must be specified.
            - context (:obj:`str`): The multi-process context of the environment manager. Default to None. \
                It can be specified as ``spawn``, ``fork`` or ``forkserver``.
            - debug (:obj:`bool`): Whether to use debug mode in the environment manager. Default to False. \
                If set True, base environment manager will be used for easy debugging. Otherwise, \
                subprocess environment manager will be used.
        """

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
            f'SQL collecting is finished, more than {n_sample} samples are collected and saved in `{save_data_path}`'
        )

    def batch_evaluate(
            self,
            env_num: int = 4,
            n_evaluator_episode: int = 4,
            context: Optional[str] = None,
            debug: bool = False
    ) -> EvalReturn:
        """
        Overview:
            Evaluate the agent with SQL algorithm for ``n_evaluator_episode`` episodes with ``env_num`` evaluator \
            environments. The evaluation result will be returned.
            The difference between methods ``batch_evaluate`` and ``deploy`` is that ``batch_evaluate`` will create \
            multiple evaluator environments to evaluate the agent to get an average performance, while ``deploy`` \
            will only create one evaluator environment to evaluate the agent and save the replay video.
        Arguments:
            - env_num (:obj:`int`): The number of evaluator environments. Default to 4.
            - n_evaluator_episode (:obj:`int`): The number of episodes to evaluate. Default to 4.
            - context (:obj:`str`): The multi-process context of the environment manager. Default to None. \
                It can be specified as ``spawn``, ``fork`` or ``forkserver``.
            - debug (:obj:`bool`): Whether to use debug mode in the environment manager. Default to False. \
                If set True, base environment manager will be used for easy debugging. Otherwise, \
                subprocess environment manager will be used.
        Returns:
            - (:obj:`EvalReturn`): The evaluation result, of which the attributions are:
                - eval_value (:obj:`np.float32`): The mean of evaluation return.
                - eval_value_std (:obj:`np.float32`): The standard deviation of evaluation return.
        """

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
    def best(self) -> 'SQLAgent':
        """
        Overview:
            Load the best model from the checkpoint directory, \
            which is by default in folder ``exp_name/ckpt/eval.pth.tar``. \
            The return value is the agent with the best model.
        Returns:
            - (:obj:`SQLAgent`): The agent with the best model.
        Examples:
            >>> agent = SQLAgent(env_id='LunarLander-v2')
            >>> agent.train()
            >>> agent = agent.best

        .. note::
            The best model is the model with the highest evaluation return. If this method is called, the current \
            model will be replaced by the best model.
        """

        best_model_file_path = os.path.join(self.checkpoint_save_dir, "eval.pth.tar")
        # Load best model if it exists
        if os.path.exists(best_model_file_path):
            policy_state_dict = torch.load(best_model_file_path, map_location=torch.device("cpu"))
            self.policy.learn_mode.load_state_dict(policy_state_dict)
        return self
