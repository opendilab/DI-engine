import os
import time
import copy
import pickle
import torch
import torch.nn as nn
from functools import partial
from tensorboardX import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from typing import Union, Optional, List, Any, Tuple, Dict

from ding.model.common.head import DiscreteHead, RegressionHead
from ding.worker import BaseLearner, BaseSerialCommander, InteractionSerialEvaluator, create_serial_collector
from ding.config import read_config, compile_config
from ding.utils import set_pkg_seed
from ding.envs import get_vec_env_setting, create_env_manager
from ding.policy.common_utils import default_preprocess_learn
from ding.policy import create_policy
from ding.utils import SequenceType, squeeze
from ding.model.common.encoder import FCEncoder, ConvEncoder


class InverseDynamicsModel(nn.Module):
    """
    InverseDynamicsModel: infering missing action information from state transition.
    input and output: given pair of observation, return action (s0,s1 --> a0 if n=2)
    """

    def __init__(
            self,
            obs_shape: Union[int, SequenceType],
            action_shape: Union[int, SequenceType],
            action_space: str,
            encoder_hidden_size_list: SequenceType = [60, 80, 100, 40],
            activation: Optional[nn.Module] = nn.LeakyReLU(),
            norm_type: Optional[str] = None
    ) -> None:
        """
        Overview:
            Init the Inverse Dynamics (encoder + head) Model according to input arguments.
        Arguments:
            - obs_shape (:obj:`Union[int, SequenceType]`): Observation space shape, such as 8 or [4, 84, 84].
            - action_shape (:obj:`Union[int, SequenceType]`): Action space shape, such as 6 or [2, 3, 3].
            - encoder_hidden_size_list (:obj:`SequenceType`): Collection of ``hidden_size`` to pass to ``Encoder``, \
                the last element must match ``head_hidden_size``.
            - activation (:obj:`Optional[nn.Module]`): The type of activation function in networks \
                if ``None`` then default set it to ``nn.LeakyReLU()`` refer to https://arxiv.org/abs/1805.01954
            - norm_type (:obj:`Optional[str]`): The type of normalization in networks, see \
                ``ding.torch_utils.fc_block`` for more details.
        """
        super(InverseDynamicsModel, self).__init__()
        # For compatibility: 1, (1, ), [4, 32, 32]
        obs_shape, action_shape = squeeze(obs_shape), squeeze(action_shape)
        # FC encoder: obs and obs[next] ,so input shape is obs_shape*2
        if isinstance(obs_shape, int) or len(obs_shape) == 1:
            self.encoder = FCEncoder(
                obs_shape * 2, encoder_hidden_size_list, activation=activation, norm_type=norm_type
            )
        elif len(obs_shape) == 3:
            obs_shape[0] = obs_shape[0] * 2
            self.encoder = ConvEncoder(obs_shape, encoder_hidden_size_list, activation=activation, norm_type=norm_type)
        else:
            raise RuntimeError(
                "not support obs_shape for pre-defined encoder: {}, please customize your own Model".format(obs_shape)
            )
        self.action_space = action_space
        assert self.action_space in ['discrete', 'continuous']
        if self.action_space == 'discrete':
            self.header = DiscreteHead(
                encoder_hidden_size_list[-1], action_shape, activation=activation, norm_type=norm_type
            )
        elif self.action_space == 'continuous':
            # self.header = ReparameterizationHead(
            #     encoder_hidden_size_list[-1],
            #     action_shape,
            #     sigma_type='conditioned',
            #     activation=activation,
            #     norm_type=norm_type
            # )
            self.header = RegressionHead(
                encoder_hidden_size_list[-1],
                action_shape,
                final_tanh=False,
                activation=activation,
                norm_type=norm_type
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.action_space == 'discrete':
            x = self.encoder(x)
            x = self.header(x)
            return x
        elif self.action_space == 'continuous':
            x = self.encoder(x)
            x = self.header(x)
            return {'logit': x['pred']}


class BCODataset(Dataset):

    def __init__(self, data=None):
        if data is None:
            self._data = []
        else:
            self._data = data

    def __len__(self):
        return len(self._data['obs'])

    def __getitem__(self, idx):
        return {k: self._data[k][idx] for k in self._data.keys()}

    @property
    def obs(self):
        return self._data['obs']

    @property
    def action(self):
        return self._data['action']


def load_expertdata(data: Dict[str, torch.Tensor]) -> None:
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


def load_agentdata(data) -> None:
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


def train_state_trainsition_model(training_set, model, n_epoch, learning_rate, action_space):
    '''
    train transition model, given pair of states return action (s0,s1 ---> a0 if n=2)
    Input:
    training_set:
    model: transition model want to train
    n: window size (how many states needed to predict the next action)
    batch_size: batch size
    n_epoch: number of epoches
    learning_rate: learning rate
    action_space: whether action is discrete or continuous
    return:
    model: trained transition model
    '''
    if action_space == "continuous":
        # criterion = nn.MSELoss()
        criterion = nn.L1Loss()
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.001)
    loss_list = []
    for itr in range(n_epoch):
        data = training_set['obs']
        y = training_set['action']
        y_pred = model.forward(data)['logit']
        loss = criterion(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())
    last_loss = loss_list[-1]
    return (model, last_loss)


def get_loss(training_set, action_space):
    if action_space == "continuous":
        # criterion = nn.MSELoss()
        criterion = nn.L1Loss()
    else:
        criterion = nn.CrossEntropyLoss()
    y = training_set['expert_action']
    y_pred = training_set['action']
    loss = criterion(y_pred, y)
    return loss.item()


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
    if cfg.policy.collect.demonstration_model_path is None:
        with open(cfg.policy.collect.demonstration_offline_data_path, 'rb') as f:
            data = pickle.load(f)
            expert_learn_dataset = load_expertdata(data)
    else:
        expert_policy = create_policy(expert_cfg.policy, model=expert_model, enable_field=['collect'])
        expert_collector_env = create_env_manager(
            expert_cfg.env.manager, [partial(env_fn, cfg=c) for c in collector_env_cfg]
        )
        expert_collector_env.seed(expert_cfg.seed)
        expert_policy.collect_mode.load_state_dict(
            torch.load(cfg.policy.collect.demonstration_model_path, map_location='cpu')
        )

        expert_collector = create_serial_collector(
            cfg.policy.collect.collector,  # for episode collector
            env=expert_collector_env,
            policy=expert_policy.collect_mode,
            exp_name=expert_cfg.exp_name
        )
        # if expert policy is sac, eps kwargs is unexpected
        if cfg.policy.action_space == "continuous":
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
        cfg.policy.model.obs_shape, cfg.policy.model.action_shape, cfg.policy.action_space,
        cfg.policy.idm_encoder_hidden_size_list
    )
    # ==========
    # Main loop
    # ==========
    learner.call_hook('before_run')
    end = time.time()
    m = nn.Softmax(dim=-1)
    collect_episode = copy.deepcopy(cfg.policy.collect.n_episode)
    agent_data = list()
    for epoch in range(cfg.policy.learn.train_epoch):
        collect_kwargs = commander.step()
        # Evaluate policy performance
        if evaluator.should_eval(learner.train_iter):
            stop, reward = evaluator.eval(learner.save_checkpoint, learner.train_iter, collector.envstep)
            if stop:
                break
        new_data = collector.collect(
            n_episode=collect_episode, train_iter=learner.train_iter, policy_kwargs=collect_kwargs
        )
        # agent_data = agent_data + new_data
        learn_dataset = load_agentdata(new_data)
        learn_dataloader = DataLoader(learn_dataset, cfg.policy.learn.idm_batch_size)
        for i, train_data in enumerate(learn_dataloader):
            (learned_model, idm_loss) = train_state_trainsition_model(
                train_data, learned_model, cfg.policy.learn.idm_train_epoch, cfg.policy.learn.idm_learning_rate,
                cfg.policy.action_space
            )
        tb_logger.add_scalar("idm_loss", idm_loss, learner.train_iter)
        # Generate state transitions from demonstrated state trajectories by IDM
        if cfg.policy.action_space == "continuous":
            expert_action_data = learned_model.forward(expert_learn_dataset.obs)['logit']
        else:
            expert_action_data = torch.argmax(m(learned_model.forward(expert_learn_dataset.obs)['logit']), -1)
        post_expert_dataset = BCODataset(
            {
                'obs': expert_learn_dataset.obs[:, 0:int(expert_learn_dataset.obs.shape[1] / 2)],
                'action': expert_action_data,
                'expert_action': expert_learn_dataset.action
            }
        )  # post_expert_dataset: Only obs and action are reserved for BC. next_obs are deleted
        expert_learn_dataloader = DataLoader(post_expert_dataset, cfg.policy.learn.batch_size)
        # Improve policy using BC
        for i, train_data in enumerate(expert_learn_dataloader):
            idm_expert_loss = get_loss(train_data, cfg.policy.action_space)
            print("idm_expert_loss:", idm_expert_loss)
            learner.data_time = time.time() - end
            learner.epoch_info = (epoch, i, len(learn_dataloader))
            learner.train(train_data, collector.envstep)
            end = time.time()
        learner.policy.get_attribute('lr_scheduler').step()
        collect_episode = int(cfg.policy.collect.n_episode * cfg.policy.collect.alpha)
        if collector.envstep >= max_env_step or learner.train_iter >= max_train_iter:
            break

    # Learner's after_run hook.
    learner.call_hook('after_run')
