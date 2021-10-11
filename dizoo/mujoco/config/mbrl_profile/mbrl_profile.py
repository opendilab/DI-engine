from typing import Union, Optional, List, Any, Tuple
import os
import copy
import torch
import logging
from functools import partial
from tensorboardX import SummaryWriter
import numpy as np
import gym

from ding.envs import get_vec_env_setting, create_env_manager, get_env_cls
from ding.model import create_model
from ding.worker import BaseLearner, InteractionSerialEvaluator, BaseSerialCommander, create_buffer, \
    create_serial_collector
from ding.config import read_config, compile_config
from ding.policy import create_policy, PolicyFactory
from ding.policy.common_utils import default_preprocess_learn
from ding.utils import set_pkg_seed, read_file, save_file
from ding.torch_utils import to_device

def save_ckpt_fn(learner, env_model, envstep):

    dirname = './{}/ckpt'.format(learner.exp_name)
    if not os.path.exists(dirname):
        try:
            os.mkdir(dirname)
        except FileExistsError:
            pass
    policy_prefix = 'policy_envstep_{}_'.format(envstep)
    model_prefix = 'model_envstep_{}_'.format(envstep)

    def model_save_ckpt_fn(ckpt_name):
        """
        Overview:
            Save checkpoint in corresponding path.
            Checkpoint info includes policy state_dict and iter num.
        Arguments:
            - engine (:obj:`BaseLearner`): the BaseLearner which needs to save checkpoint
        """
        path = os.path.join(dirname, ckpt_name)
        state_dict = env_model.state_dict()
        save_file(path, state_dict)
        learner.info('env model save ckpt in {}'.format(path))

    def model_policy_save_ckpt_fn(ckpt_name):
        model_save_ckpt_fn(model_prefix + ckpt_name)
        learner.save_checkpoint(policy_prefix + ckpt_name)

    return model_policy_save_ckpt_fn

def create_model_env(cfg):
    cfg = copy.deepcopy(cfg)
    model_env_fn = get_env_cls(cfg)
    cfg.pop('import_names')
    cfg.pop('type')
    return model_env_fn(**cfg)

def get_dist_and_q(policy, obs, n_actions):
    with torch.no_grad():
        policy._learn_model.train()
        # get distributions
        (mu, sigma) = policy._learn_model.forward(obs, mode='compute_actor')['logit']
        # get uniform actions
        b, a, s = mu.shape[0], mu.shape[1], obs.shape[-1]
        action = mu.tanh().view(b, 1, a).repeat(1, a * (n_actions + 1), 1).view(b, a, n_actions + 1, a) # [b, a, n_actions + 1, a]
        act_index = torch.arange(a).to(mu.device)
        uni_action = torch.arange(n_actions + 1).to(mu.device).float() / n_actions * 2 - 1 # to evenly divided in [-1,1]
        action[:,act_index,:,act_index] = uni_action
        action = action.view(b * a * (n_actions + 1), a)
        obs = obs.view(b, 1, s).repeat(1, a * (n_actions + 1), 1).view(b * a * (n_actions + 1), s)
        # get uniform actions' q_values
        q = policy._learn_model.forward({'obs': obs, 'action': action}, mode='compute_critic')['q_value']
        if policy._twin_critic:
            q = torch.min(q[0],q[1])
        q = q.view(b, a, n_actions + 1)
    return mu, sigma, q, obs, action

def get_model_pred(policy, env_model, obs, action):
    with torch.no_grad():
        # predict next actions
        reward, next_obs = env_model.batch_predict(obs, action)
        # get predicted next q_values
        policy._learn_model.train()
        (mu, sigma) = policy._learn_model.forward(next_obs, mode='compute_actor')['logit']
        next_q = policy._learn_model.forward({'obs': next_obs, 'action': mu.tanh()}, mode='compute_critic')['q_value']
        if policy._twin_critic:
            next_q = torch.min(next_q[0], next_q[1])
    return reward, next_obs, next_q

def get_env_step(policy, env, obs, action):
    b = len(obs)
    obs = torch.cat([torch.zeros(b, 1).to(obs), obs], dim=1).view(b, 2, -1).contiguous()
    obs, action = obs.cpu().numpy(), action.cpu().numpy()
    reward, next_obs = [], []
    for i, (o, a) in enumerate(zip(obs, action)):
        env.reset()
        env.set_state(o[0], o[1])
        n, r, d, _ = env.step(a)
        reward.append(r)
        next_obs.append(n)
        if (i + 1) % 1000 == 0:
            print('eval the {}-th sample'.format(i))
    reward = torch.from_numpy(np.stack(reward)).float().cuda()
    next_obs = torch.from_numpy(np.stack(next_obs)).float().cuda()
    with torch.no_grad():
        # get predicted next q_values
        policy._learn_model.train()
        (mu, sigma) = policy._learn_model.forward(next_obs, mode='compute_actor')['logit']
        next_q = policy._learn_model.forward({'obs': next_obs, 'action': mu.tanh()}, mode='compute_critic')['q_value']
        if policy._twin_critic:
            next_q = torch.min(next_q[0],next_q[1])
    return reward, next_obs, next_q

def profile(
        input_cfg: Union[str, Tuple[dict, dict]],
        policy_path_1,
        policy_path_2,
        model_path,
        exp_name='profile',
        seed: int = 0,
        n_samples = 1000,
        n_actions = 20,
) -> 'Policy':  # noqa
    """
    Overview:
        Serial pipeline entry.
    Arguments:
        - input_cfg (:obj:`Union[str, Tuple[dict, dict]]`): Config in dict type. \
            ``str`` type means config file path. \
            ``Tuple[dict, dict]`` type means [user_config, create_cfg].
        - seed (:obj:`int`): Random seed.
        - env_setting (:obj:`Optional[List[Any]]`): A list with 3 elements: \
            ``BaseEnv`` subclass, collector env config, and evaluator env config.
        - model (:obj:`Optional[torch.nn.Module]`): Instance of torch.nn.Module.
        - max_iterations (:obj:`Optional[torch.nn.Module]`): Learner's max iteration. Pipeline will stop \
            when reaching this iteration.
    Returns:
        - policy (:obj:`Policy`): Converged policy.
    """
    # Compile config
    if isinstance(input_cfg, str):
        cfg, create_cfg = read_config(input_cfg)
    else:
        cfg, create_cfg = input_cfg
    model_based_cfg = cfg.pop('model_based')
    create_cfg.policy.type = create_cfg.policy.type + '_command'
    cfg = compile_config(cfg, seed=seed, env=None, auto=True, create_cfg=create_cfg, save_cfg=True)

    # Create logger
    tb_logger = SummaryWriter(os.path.join('./{}/log/'.format(cfg.exp_name), 'serial'))

    # Create env
    env_fn, collector_env_cfg, evaluator_env_cfg = get_vec_env_setting(cfg.env)

    collector_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in collector_env_cfg])
    evaluator_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in evaluator_env_cfg])
    collector_env.seed(cfg.seed)
    evaluator_env.seed(cfg.seed, dynamic_seed=False)

    # Create env model
    model_based_cfg.env_model.tb_logger = tb_logger
    env_model = create_model(model_based_cfg.env_model)
    env_model.load_state_dict(read_file(model_path))

    # Create policy
    set_pkg_seed(cfg.seed, use_cuda=cfg.policy.cuda)
    policy = create_policy(cfg.policy, model=None, enable_field=['learn', 'collect', 'eval', 'command'])
    policy._load_state_dict_learn(read_file(policy_path_1))

    # Create worker components: collector, evaluator, commander.
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

    # ==========
    # Main loop
    # ==========

    # Accumulate plenty of data for of profiling.
    print('collecting data')
    data = collector.collect(n_sample=n_samples, policy_kwargs={})
    data = default_preprocess_learn(
        data,
        use_priority=False,
        use_priority_IS_weight=False,
        ignore_done=False,
        use_nstep=False
    )
    if policy._cuda:
        data = to_device(data, policy._device)
    obs = data.get('obs')
    done = data.get('done').bool()
    obs = obs[~done]

    # Profiling
    print('getting distributions and q values')
    mu, sigma, q, obs_, action = get_dist_and_q(policy, obs, n_actions)
    print('getting moel predictions')
    reward_pred, next_obs_pred, next_q_pred = get_model_pred(policy, env_model, obs_, action)
    print('getting real steps')
    reward_real, next_obs_real, next_q_real = get_env_step(policy, gym.make(cfg.env.env_id), obs_, action)
    policy._load_state_dict_learn(read_file(policy_path_2))
    print('getting distributions and q values with new policy')
    mu_new, sigma_new, q_new, _, _ = get_dist_and_q(policy, obs, n_actions)

    # Save
    b, a, s = mu.shape[0], mu.shape[1], obs.shape[1]
    reward_pred = reward_pred.view(b, a, n_actions + 1)
    reward_real = reward_real.view(b, a, n_actions + 1)
    next_obs_pred = next_obs_pred.view(b, a, n_actions + 1, s)
    next_obs_real = next_obs_real.view(b, a, n_actions + 1, s)
    next_q_pred = next_q_pred.view(b, a, n_actions + 1)
    next_q_real = next_q_real.view(b, a, n_actions + 1)
    dict = {'obs':obs, 'dist':(mu, sigma), 'dist_new':(mu_new, sigma_new), 'q':q, 'q_new':q_new, 'next_q_pred':next_q_pred, 'next_q_real':next_q_real, 'next_obs_pred':next_obs_pred, 'next_obs_real':next_obs_real, 'reward_pred':reward_pred, 'reward_real':reward_real}
    save_file(exp_name+'.pth.tar', dict)


envstep_1 = 25500
envstep_2 = 48750
exp_name = 'sac_hopper_mopo_default_config'
n_samples = 1000
n_actions = 20

root_dir = '../'
exp_dir = exp_name + '/'
exp_name = exp_name + '_{}_{}'.format(envstep_1, envstep_2)
ckpt_dir = 'default_experiment/ckpt/'
dir = root_dir + exp_dir + ckpt_dir

policy_ckpt_1 = 'policy_envstep_{}_ckpt_best.pth.tar'.format(envstep_1)
policy_ckpt_2 = 'policy_envstep_{}_ckpt_best.pth.tar'.format(envstep_2)
model_ckpt = 'model_envstep_{}_ckpt_best.pth.tar'.format(envstep_1)
policy_path_1 = dir + policy_ckpt_1
policy_path_2 = dir + policy_ckpt_2
model_path = dir + model_ckpt

from dizoo.mujoco.config.sac_hopper_mopo_default_config.config import main_config, create_config

profile(
        (main_config, create_config),
        policy_path_1,
        policy_path_2,
        model_path,
        exp_name=exp_name,
        seed=0,
        n_samples = n_samples,
        n_actions = n_actions,
)
    # # Train
    # batch_size = learner.policy.get_attribute('batch_size')
    # real_ratio = model_based_cfg['real_ratio']
    # replay_batch_size = int(batch_size*real_ratio)
    # imagine_batch_size = batch_size - replay_batch_size
    # eval_buffer = []
    # for _ in range(max_iterations):
    #     collect_kwargs = commander.step()
    #     # Evaluate policy performance
    #     if evaluator.should_eval(learner.train_iter):
    #         stop, reward = evaluator.eval(save_ckpt_fn(learner, env_model, collector.envstep), learner.train_iter, collector.envstep)
    #         if stop:
    #             break
    #     # Collect data by default config n_sample/n_episode
    #     new_data = collector.collect(train_iter=learner.train_iter, policy_kwargs=collect_kwargs)
    #     replay_buffer.push(new_data, cur_collector_envstep=collector.envstep)
    #     eval_buffer.extend(new_data)
    #     # Eval env_model
    #     if env_model.should_eval(collector.envstep):
    #         env_model.eval(eval_buffer, collector.envstep)
    #         eval_buffer = []
    #     # Train env_model and use model_env to rollout
    #     if env_model.should_train(collector.envstep):
    #         env_model.train(replay_buffer, learner.train_iter, collector.envstep)
    #         imagine_buffer.update(collector.envstep)
    #         model_env.rollout(env_model, policy.collect_mode, replay_buffer, imagine_buffer, collector.envstep, learner.train_iter)
    #         policy._rollout_length = model_env._set_rollout_length(collector.envstep)
    #     # Learn policy from collected data
    #     for i in range(cfg.policy.learn.update_per_collect):
    #         # Learner will train ``update_per_collect`` times in one iteration.
    #         replay_train_data = replay_buffer.sample(replay_batch_size, learner.train_iter)
    #         imagine_batch_data = imagine_buffer.sample(imagine_batch_size, learner.train_iter)
    #         if replay_train_data is None or imagine_batch_data is None:
    #             break
    #         train_data = replay_train_data + imagine_batch_data
    #         learner.train(train_data, collector.envstep)
    #         # Priority is not support
    #         # if learner.policy.get_attribute('priority'):
    #         #     replay_buffer.update(learner.priority_info)
    #     if cfg.policy.on_policy:
    #         # On-policy algorithm must clear the replay buffer.
    #         replay_buffer.clear()
    #         imagine_buffer.clear()
