import os
import pickle
import torch
from functools import partial
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from typing import Union, Optional, List, Any, Tuple, Dict

from ding.worker import BaseLearner, BaseSerialCommander, InteractionSerialEvaluator, create_serial_collector
from ding.config import read_config, compile_config
from ding.utils import set_pkg_seed
from ding.envs import get_vec_env_setting, create_env_manager
from ding.policy.common_utils import default_preprocess_learn
from ding.policy import create_policy
from ding.utils.data.dataset import BCODataset
from ding.world_model.idm import InverseDynamicsModel


def load_expertdata(data: Dict[str, torch.Tensor]) -> BCODataset:
    """
    loading from demonstration data, which only have obs and next_obs
    action need to be inferred from Inverse Dynamics Model
    """
    post_data = list()
    for episode in range(len(data)):
        for transition in data[episode]:
            transition['episode_id'] = episode
            post_data.append(transition)
    post_data = default_preprocess_learn(post_data)
    return BCODataset(
        {
            'obs': torch.cat((post_data['obs'], post_data['next_obs']), 1),
            'episode_id': post_data['episode_id'],
            'action': post_data['action']
        }
    )


def load_agentdata(data) -> BCODataset:
    """
    loading from policy data, which only have obs and next_obs as features and action as label
    """
    post_data = list()
    for episode in range(len(data)):
        for transition in data[episode]:
            transition['episode_id'] = episode
            post_data.append(transition)
    post_data = default_preprocess_learn(post_data)
    return BCODataset(
        {
            'obs': torch.cat((post_data['obs'], post_data['next_obs']), 1),
            'action': post_data['action'],
            'episode_id': post_data['episode_id']
        }
    )


def serial_pipeline_bco(
        input_cfg: Union[str, Tuple[dict, dict]],
        expert_cfg: Union[str, Tuple[dict, dict]],
        seed: int = 0,
        env_setting: Optional[List[Any]] = None,
        model: Optional[torch.nn.Module] = None,
        expert_model: Optional[torch.nn.Module] = None,
        # model: Optional[torch.nn.Module] = None,
        max_train_iter: Optional[int] = int(1e10),
        max_env_step: Optional[int] = int(1e10),
) -> None:

    if isinstance(input_cfg, str):
        cfg, create_cfg = read_config(input_cfg)
        expert_cfg, expert_create_cfg = read_config(expert_cfg)
    else:
        cfg, create_cfg = input_cfg
        expert_cfg, expert_create_cfg = expert_cfg
    create_cfg.policy.type = create_cfg.policy.type + '_command'
    expert_create_cfg.policy.type = expert_create_cfg.policy.type + '_command'
    env_fn = None if env_setting is None else env_setting[0]
    cfg = compile_config(cfg, seed=seed, env=env_fn, auto=True, create_cfg=create_cfg, save_cfg=True)
    expert_cfg = compile_config(
        expert_cfg, seed=seed, env=env_fn, auto=True, create_cfg=expert_create_cfg, save_cfg=True
    )
    # Random seed
    set_pkg_seed(cfg.seed, use_cuda=cfg.policy.cuda)
    # Create main components: env, policy
    if env_setting is None:
        env_fn, collector_env_cfg, evaluator_env_cfg = get_vec_env_setting(cfg.env)
    else:
        env_fn, collector_env_cfg, evaluator_env_cfg = env_setting

    # Generate Expert Data
    if cfg.policy.collect.model_path is None:
        with open(cfg.policy.collect.data_path, 'rb') as f:
            data = pickle.load(f)
            expert_learn_dataset = load_expertdata(data)
    else:
        expert_policy = create_policy(expert_cfg.policy, model=expert_model, enable_field=['collect'])
        expert_collector_env = create_env_manager(
            expert_cfg.env.manager, [partial(env_fn, cfg=c) for c in collector_env_cfg]
        )
        expert_collector_env.seed(expert_cfg.seed)
        expert_policy.collect_mode.load_state_dict(torch.load(cfg.policy.collect.model_path, map_location='cpu'))

        expert_collector = create_serial_collector(
            cfg.policy.collect.collector,  # for episode collector
            env=expert_collector_env,
            policy=expert_policy.collect_mode,
            exp_name=expert_cfg.exp_name
        )
        # if expert policy is sac, eps kwargs is unexpected
        if cfg.policy.continuous:
            expert_data = expert_collector.collect(n_episode=100)
        else:
            policy_kwargs = {'eps': 0}
            expert_data = expert_collector.collect(n_episode=100, policy_kwargs=policy_kwargs)
        expert_learn_dataset = load_expertdata(expert_data)
        expert_collector.reset_policy(expert_policy.collect_mode)

    # Main components
    tb_logger = SummaryWriter(os.path.join('./{}/log/'.format(cfg.exp_name), 'serial'))
    policy = create_policy(cfg.policy, model=model, enable_field=['learn', 'collect', 'eval', 'command'])
    learner = BaseLearner(cfg.policy.learn.learner, policy.learn_mode, tb_logger, exp_name=cfg.exp_name)
    collector_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in collector_env_cfg])
    evaluator_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in evaluator_env_cfg])
    collector_env.seed(cfg.seed)
    evaluator_env.seed(cfg.seed, dynamic_seed=False)
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
    commander = BaseSerialCommander(
        cfg.policy.other.commander, learner, collector, evaluator, None, policy=policy.command_mode
    )
    learned_model = InverseDynamicsModel(
        cfg.policy.model.obs_shape, cfg.policy.model.action_shape, cfg.bco.model.idm_encoder_hidden_size_list,
        cfg.bco.model.action_space
    )
    # ==========
    # Main loop
    # ==========
    learner.call_hook('before_run')
    collect_episode = int(cfg.policy.collect.n_episode * cfg.bco.alpha)
    init_episode = True
    while True:
        collect_kwargs = commander.step()
        # Evaluate policy performance
        if evaluator.should_eval(learner.train_iter):
            stop, reward = evaluator.eval(learner.save_checkpoint, learner.train_iter, collector.envstep)
            if stop:
                break

        if init_episode:
            new_data = collector.collect(
                n_episode=cfg.policy.collect.n_episode, train_iter=learner.train_iter, policy_kwargs=collect_kwargs
            )
            init_episode = False
        else:
            new_data = collector.collect(
                n_episode=collect_episode, train_iter=learner.train_iter, policy_kwargs=collect_kwargs
            )
        learn_dataset = load_agentdata(new_data)
        learn_dataloader = DataLoader(learn_dataset, cfg.bco.learn.idm_batch_size)
        for i, train_data in enumerate(learn_dataloader):
            idm_loss = learned_model.train(
                train_data,
                cfg.bco.learn.idm_train_epoch,
                cfg.bco.learn.idm_learning_rate,
                cfg.bco.learn.idm_weight_decay,
            )
        # tb_logger.add_scalar("learner_iter/idm_loss", idm_loss, learner.train_iter)
        # tb_logger.add_scalar("learner_step/idm_loss", idm_loss, collector.envstep)
        # Generate state transitions from demonstrated state trajectories by IDM
        expert_action_data = learned_model.predict_action(expert_learn_dataset.obs)['action']
        post_expert_dataset = BCODataset(
            {
                # next_obs are deleted
                'obs': expert_learn_dataset.obs[:, 0:int(expert_learn_dataset.obs.shape[1] // 2)],
                'action': expert_action_data,
                'expert_action': expert_learn_dataset.action
            }
        )  # post_expert_dataset: Only obs and action are reserved for BC. next_obs are deleted
        expert_learn_dataloader = DataLoader(post_expert_dataset, cfg.policy.learn.batch_size)
        # Improve policy using BC
        for epoch in range(cfg.policy.learn.train_epoch):
            for i, train_data in enumerate(expert_learn_dataloader):
                learner.train(train_data, collector.envstep)
            if cfg.policy.learn.lr_decay:
                learner.policy.get_attribute('lr_scheduler').step()
        if collector.envstep >= max_env_step or learner.train_iter >= max_train_iter:
            break

    # Learner's after_run hook.
    learner.call_hook('after_run')
