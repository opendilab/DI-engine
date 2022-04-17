"""The code is adapted from https://github.com/nikhilbarhate99/min-decision-transformer
"""

from typing import List, Dict, Any, Tuple, Union
from collections import namedtuple
import copy
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal, Independent

from ding.torch_utils import Adam, to_device
from ding.rl_utils import v_1step_td_data, v_1step_td_error, get_train_sample, \
    qrdqn_nstep_td_data, qrdqn_nstep_td_error, get_nstep_return_data
from ding.model import model_wrap
from ding.utils import POLICY_REGISTRY
from ding.utils.data import default_collate, default_decollate
from .sac import SACPolicy
from .dqn import DQNPolicy
from .common_utils import default_preprocess_learn
from ding.model.template.decision_transformer import DecisionTransformer
import os
from datetime import datetime
import gym
from ding.torch_utils import one_hot
import csv

## from infos.py from official d4rl github repo
REF_MIN_SCORE = {
    'halfcheetah': -280.178953,
    'walker2d': 1.629008,
    'hopper': -20.272305,
}

REF_MAX_SCORE = {
    'halfcheetah': 12135.0,
    'walker2d': 4592.3,
    'hopper': 3234.3,
}

## calculated from d4rl datasets
D4RL_DATASET_STATS = {
    'halfcheetah-medium-v2': {
        'state_mean': [
            -0.06845773756504059, 0.016414547339081764, -0.18354906141757965, -0.2762460708618164, -0.34061527252197266,
            -0.09339715540409088, -0.21321271359920502, -0.0877423882484436, 5.173007488250732, -0.04275195300579071,
            -0.036108363419771194, 0.14053793251514435, 0.060498327016830444, 0.09550975263118744, 0.06739100068807602,
            0.005627387668937445, 0.013382787816226482
        ],
        'state_std': [
            0.07472999393939972, 0.3023499846458435, 0.30207309126853943, 0.34417077898979187, 0.17619241774082184,
            0.507205605506897, 0.2567007839679718, 0.3294812738895416, 1.2574149370193481, 0.7600541710853577,
            1.9800915718078613, 6.565362453460693, 7.466367721557617, 4.472222805023193, 10.566964149475098,
            5.671932697296143, 7.4982590675354
        ]
    },
    'halfcheetah-medium-replay-v2': {
        'state_mean': [
            -0.12880703806877136, 0.3738119602203369, -0.14995987713336945, -0.23479078710079193, -0.2841278612613678,
            -0.13096535205841064, -0.20157982409000397, -0.06517726927995682, 3.4768247604370117, -0.02785065770149231,
            -0.015035249292850494, 0.07697279006242752, 0.01266712136566639, 0.027325302362442017, 0.02316424623131752,
            0.010438721626996994, -0.015839405357837677
        ],
        'state_std': [
            0.17019015550613403, 1.284424901008606, 0.33442774415016174, 0.3672759234905243, 0.26092398166656494,
            0.4784106910228729, 0.3181420564651489, 0.33552637696266174, 2.0931615829467773, 0.8037433624267578,
            1.9044333696365356, 6.573209762573242, 7.572863578796387, 5.069749355316162, 9.10555362701416,
            6.085654258728027, 7.25300407409668
        ]
    },
    'halfcheetah-medium-expert-v2': {
        'state_mean': [
            -0.05667462572455406, 0.024369969964027405, -0.061670560389757156, -0.22351515293121338,
            -0.2675151228904724, -0.07545716315507889, -0.05809682980179787, -0.027675075456500053, 8.110626220703125,
            -0.06136331334710121, -0.17986927926540375, 0.25175222754478455, 0.24186332523822784, 0.2519369423389435,
            0.5879552960395813, -0.24090635776519775, -0.030184272676706314
        ],
        'state_std': [
            0.06103534251451492, 0.36054104566574097, 0.45544400811195374, 0.38476887345314026, 0.2218363732099533,
            0.5667523741722107, 0.3196682929992676, 0.2852923572063446, 3.443821907043457, 0.6728139519691467,
            1.8616976737976074, 9.575807571411133, 10.029894828796387, 5.903450012207031, 12.128185272216797,
            6.4811787605285645, 6.378620147705078
        ]
    },
    'walker2d-medium-v2': {
        'state_mean': [
            1.218966007232666, 0.14163373410701752, -0.03704913705587387, -0.13814310729503632, 0.5138224363327026,
            -0.04719110205769539, -0.47288352251052856, 0.042254164814949036, 2.3948874473571777, -0.03143199160695076,
            0.04466355964541435, -0.023907244205474854, -0.1013401448726654, 0.09090937674045563, -0.004192637279629707,
            -0.12120571732521057, -0.5497063994407654
        ],
        'state_std': [
            0.12311358004808426, 0.3241879940032959, 0.11456084251403809, 0.2623065710067749, 0.5640279054641724,
            0.2271878570318222, 0.3837319612503052, 0.7373676896095276, 1.2387926578521729, 0.798020601272583,
            1.5664079189300537, 1.8092705011367798, 3.025604248046875, 4.062486171722412, 1.4586567878723145,
            3.7445690631866455, 5.5851287841796875
        ]
    },
    'walker2d-medium-replay-v2': {
        'state_mean': [
            1.209364652633667, 0.13264022767543793, -0.14371201395988464, -0.2046516090631485, 0.5577612519264221,
            -0.03231537342071533, -0.2784661054611206, 0.19130706787109375, 1.4701707363128662, -0.12504704296588898,
            0.0564953051507473, -0.09991033375263214, -0.340340256690979, 0.03546293452382088, -0.08934258669614792,
            -0.2992438077926636, -0.5984178185462952
        ],
        'state_std': [
            0.11929835379123688, 0.3562574088573456, 0.25852200388908386, 0.42075422406196594, 0.5202291011810303,
            0.15685082972049713, 0.36770978569984436, 0.7161387801170349, 1.3763766288757324, 0.8632221817970276,
            2.6364643573760986, 3.0134117603302, 3.720684051513672, 4.867283821105957, 2.6681625843048096,
            3.845186948776245, 5.4768385887146
        ]
    },
    'walker2d-medium-expert-v2': {
        'state_mean': [
            1.2294334173202515, 0.16869689524173737, -0.07089081406593323, -0.16197483241558075, 0.37101927399635315,
            -0.012209027074277401, -0.42461398243904114, 0.18986578285694122, 3.162475109100342, -0.018092676997184753,
            0.03496946766972542, -0.013921679928898811, -0.05937029421329498, -0.19549426436424255,
            -0.0019200450042262673, -0.062483321875333786, -0.27366524934768677
        ],
        'state_std': [
            0.09932824969291687, 0.25981399416923523, 0.15062759816646576, 0.24249176681041718, 0.6758718490600586,
            0.1650741547346115, 0.38140663504600525, 0.6962361335754395, 1.3501490354537964, 0.7641991376876831,
            1.534574270248413, 2.1785972118377686, 3.276582717895508, 4.766193866729736, 1.1716983318328857,
            4.039782524108887, 5.891613960266113
        ]
    },
    'hopper-medium-v2': {
        'state_mean': [
            1.311279058456421, -0.08469521254301071, -0.5382719039916992, -0.07201576232910156, 0.04932365566492081,
            2.1066856384277344, -0.15017354488372803, 0.008783451281487942, -0.2848185896873474, -0.18540096282958984,
            -0.28461286425590515
        ],
        'state_std': [
            0.17790751159191132, 0.05444620922207832, 0.21297138929367065, 0.14530418813228607, 0.6124444007873535,
            0.8517446517944336, 1.4515252113342285, 0.6751695871353149, 1.5362390279769897, 1.616074562072754,
            5.607253551483154
        ]
    },
    'hopper-medium-replay-v2': {
        'state_mean': [
            1.2305138111114502, -0.04371410980820656, -0.44542956352233887, -0.09370097517967224, 0.09094487875699997,
            1.3694725036621094, -0.19992674887180328, -0.022861352190375328, -0.5287045240402222, -0.14465883374214172,
            -0.19652697443962097
        ],
        'state_std': [
            0.1756512075662613, 0.0636928603053093, 0.3438323438167572, 0.19566889107227325, 0.5547984838485718,
            1.051029920578003, 1.158307671546936, 0.7963128685951233, 1.4802359342575073, 1.6540331840515137,
            5.108601093292236
        ]
    },
    'hopper-medium-expert-v2': {
        'state_mean': [
            1.3293815851211548, -0.09836531430482864, -0.5444297790527344, -0.10201650857925415, 0.02277466468513012,
            2.3577215671539307, -0.06349576264619827, -0.00374026270583272, -0.1766270101070404, -0.11862941086292267,
            -0.12097819894552231
        ],
        'state_std': [
            0.17012375593185425, 0.05159067362546921, 0.18141433596611023, 0.16430604457855225, 0.6023368239402771,
            0.7737284898757935, 1.4986555576324463, 0.7483318448066711, 1.7953159809112549, 2.0530025959014893,
            5.725032806396484
        ]
    },
}


@POLICY_REGISTRY.register('dt')
class DTPolicy(DQNPolicy):
    r"""
    Overview:
        Policy class of DT algorithm in discrete environments.
    """
    config = dict(
        # (str) RL policy register name (refer to function "POLICY_REGISTRY").
        type='dt',
        # (bool) Whether to use cuda for network.
        cuda=False,
        # (bool) Whether the RL algorithm is on-policy or off-policy.
        on_policy=False,
        # (bool) Whether use priority(priority sample, IS weight, update priority)
        priority=False,
        # (float) Reward's future discount factor, aka. gamma.
        discount_factor=0.97,
        # (int) N-step reward for target q_value estimation
        nstep=1,
        obs_shape=4,
        action_shape=2,
        # encoder_hidden_size_list=[128, 128, 64],
        dataset='medium',  # medium / medium-replay / medium-expert
        rtg_scale=1000,  # normalize returns to go
        max_eval_ep_len=1000,  # max len of one episode
        num_eval_ep=10,  # num of evaluation episodes
        batch_size=64,  # training batch size
        lr=1e-4,
        wt_decay=1e-4,
        warmup_steps=10000,
        max_train_iters=200,
        num_updates_per_iter=100,
        context_len=20,
        n_blocks=3,
        embed_dim=128,
        n_heads=1,
        dropout_p=0.1,
        learn=dict(
            # (bool) Whether to use multi gpu
            multi_gpu=False,
            # How many updates(iterations) to train after collector's one collection.
            # Bigger "update_per_collect" means bigger off-policy.
            # collect data -> update policy-> collect data -> ...
            update_per_collect=1,
            # batch_size=64,
            learning_rate=0.001,
            # ==============================================================
            # The following configs are algorithm-specific
            # ==============================================================
            # (int) Frequence of target network update.
            target_update_freq=100,
            # (bool) Whether ignore done(usually for max step termination env)
            ignore_done=False,
            # (float) Loss weight for conservative item.
            min_q_weight=1.0,
        ),
        # collect_mode config
        collect=dict(
            # (int) Cut trajectories into pieces with length "unroll_len".
            unroll_len=1,
        ),
        eval=dict(),
        # other config
        other=dict(
            # Epsilon greedy with decay.
            eps=dict(
                # (str) Decay type. Support ['exp', 'linear'].
                type='exp',
                start=0.95,
                end=0.1,
                # (int) Decay length(env step)
                decay=10000,
            ),
            replay_buffer=dict(replay_buffer_size=10000, )
        ),
    )

    def _init_learn(self) -> None:
        r"""
            Overview:
                Learn mode init method. Called by ``self.__init__``.
                Init the optimizer, algorithm config, main and target models.
            """

        if self._cfg.env_name == 'walker2d':
            env_name = 'Walker2d-v3'
            rtg_target = 5000
            env_d4rl_name = f'walker2d-{dataset}-v2'

        elif self._cfg.env_name == 'halfcheetah':
            env_name = 'HalfCheetah-v3'
            rtg_target = 6000
            env_d4rl_name = f'halfcheetah-{dataset}-v2'

        elif self._cfg.env_name == 'hopper':
            env_name = 'Hopper-v3'
            rtg_target = 3600
            env_d4rl_name = f'hopper-{dataset}-v2'

        else:
            # raise NotImplementedError
            rtg_target = 200
            self.env_name = 'CartPole-v0'

        self.stop_value = self._cfg.stop_value
        self.env_name = self._cfg.env_name
        dataset = self._cfg.dataset  # medium / medium-replay / medium-expert
        self.rtg_scale = self._cfg.rtg_scale  # normalize returns to go
        self.max_test_ep_len = self._cfg.max_test_ep_len
        self.rtg_target = rtg_target
        self.max_eval_ep_len = self._cfg.max_eval_ep_len  # max len of one episode
        self.num_eval_ep = self._cfg.num_eval_ep  # num of evaluation episodes

        # self.batch_size = self._cfg.learn.batch_size            # training batch size
        lr = self._cfg.lr  # learning rate
        wt_decay = self._cfg.wt_decay  # weight decay
        warmup_steps = self._cfg.warmup_steps  # warmup steps for lr scheduler

        # total updates = max_train_iters x num_updates_per_iter
        max_train_iters = self._cfg.max_train_iters
        self.num_updates_per_iter = self._cfg.num_updates_per_iter

        self.context_len = self._cfg.context_len  # K in decision transformer
        n_blocks = self._cfg.n_blocks  # num of transformer blocks
        embed_dim = self._cfg.embed_dim  # embedding (hidden) dim of transformer
        n_heads = self._cfg.n_heads  # num of transformer heads
        dropout_p = self._cfg.dropout_p  # dropout probability

        # # load data from this file
        # dataset_path = f'{self._cfg.dataset_dir}/{env_d4rl_name}.pkl'

        # saves model and csv in this directory
        self.log_dir = self._cfg.log_dir
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # training and evaluation device
        self.device = torch.device(self._cfg.device)
        device = torch.device(self._cfg.device)

        self.start_time = datetime.now().replace(microsecond=0)
        self.start_time_str = self.start_time.strftime("%y-%m-%d-%H-%M-%S")

        # prefix = "dt_" + env_d4rl_name
        self.prefix = "dt_" + self.env_name

        save_model_name = self.prefix + "_model_" + self.start_time_str + ".pt"
        self.save_model_path = os.path.join(self.log_dir, save_model_name)
        self.save_best_model_path = self.save_model_path[:-3] + "_best.pt"

        log_csv_name = self.prefix + "_log_" + self.start_time_str + ".csv"
        log_csv_path = os.path.join(self.log_dir, log_csv_name)

        # self.csv_writer = csv.writer(open(log_csv_path, 'a', 1))
        # csv_header = (["duration", "num_updates", "action_loss",
        #             "eval_avg_reward", "eval_avg_ep_len", "eval_d4rl_score"])

        # self.csv_writer.writerow(csv_header)

        dataset_path = self._cfg.learn.dataset_path
        print("=" * 60)
        print("start time: " + self.start_time_str)
        print("=" * 60)

        print("device set to: " + str(device))
        print("dataset path: " + dataset_path)
        print("model save path: " + self.save_model_path)
        print("log csv save path: " + log_csv_path)

        # traj_dataset = D4RLTrajectoryDataset(dataset_path, self.context_len, rtg_scale)

        # traj_data_loader = DataLoader(
        #                         traj_dataset,
        #                         batch_size=batch_size,
        #                         shuffle=True,
        #                         pin_memory=True,
        #                         drop_last=True
        #                     )

        # data_iter = iter(traj_data_loader)

        # ## get state stats from dataset
        # self.state_mean, self.state_std = traj_dataset.get_state_stats()

        self._env = gym.make(self.env_name)

        # state_dim = env.observation_space.shape[0]
        # act_dim = env.action_space.shape[0]
        self.state_dim = self._cfg.model.state_dim
        self.act_dim = self._cfg.model.act_dim

        # self._learn_model = DecisionTransformer(
        #             state_dim=self._cfg.model.state_dim,
        #             act_dim=self._cfg.model.act_dim,
        #             n_blocks=self._cfg.model.n_blocks,
        #             h_dim=self._cfg.model.embed_dim,
        #             context_len=self._cfg.model.context_len,
        #             n_heads=self._cfg.model.n_heads,
        #             drop_p=self._cfg.model.dropout_p,
        #         ).to(device)

        self._learn_model = self._model
        self._optimizer = torch.optim.AdamW(self._learn_model.parameters(), lr=lr, weight_decay=wt_decay)

        self._scheduler = torch.optim.lr_scheduler.LambdaLR(
            self._optimizer, lambda steps: min((steps + 1) / warmup_steps, 1)
        )

        self.max_d4rl_score = -1.0
        self.max_cartpole_score = -1.0

        self.total_updates = 0

    def _forward_learn(self, data: dict) -> Dict[str, Any]:
        r"""
            Overview:
                Forward and backward function of learn mode.
            Arguments:
                - data (:obj:`dict`): Dict type data, including at least ['obs', 'action', 'reward', 'next_obs']
            Returns:
                - info_dict (:obj:`Dict[str, Any]`): Including current lr and loss.
        """
        data_iter = data['data_iter']
        traj_data_loader = data['traj_data_loader']

        self.log_action_losses = []
        self._learn_model.train()
        self._learn_model.to(self.device)  #TODO(pu)

        for _ in range(self.num_updates_per_iter):
            try:
                timesteps, states, actions, returns_to_go, traj_mask = next(data_iter)
            except StopIteration:
                data_iter = iter(traj_data_loader)
                timesteps, states, actions, returns_to_go, traj_mask = next(data_iter)

            timesteps = timesteps.to(self.device)  # B x T
            states = states.to(self.device)  # B x T x state_dim
            actions = actions.to(self.device)  # B x T x act_dim
            returns_to_go = returns_to_go.to(self.device).unsqueeze(dim=-1)  # B x T x 1
            traj_mask = traj_mask.to(self.device)  # B x T
            action_target = torch.clone(actions).detach().to(self.device)

            # if discrete
            if not self._cfg.model.continuous:
                actions = one_hot(actions.squeeze(-1), num=self.act_dim)
                returns_to_go = returns_to_go.to(self.device).squeeze(dim=-1)  # B x T x 1

            returns_to_go =  returns_to_go.float()
            state_preds, action_preds, return_preds = self._learn_model.forward(
                timesteps=timesteps, states=states, actions=actions, returns_to_go=returns_to_go
            )
            # only consider non padded elements
            action_preds = action_preds.view(-1, self.act_dim)[traj_mask.view(-1, ) > 0]

            if self._cfg.model.continuous:
                action_target = action_target.view(-1, self.act_dim)[traj_mask.view(-1, ) > 0]
            else:
                action_target = action_target.view(-1)[traj_mask.view(-1, ) > 0]

            if self._cfg.model.continuous:
                action_loss = F.mse_loss(action_preds, action_target, reduction='mean')
            else:
                action_loss = F.cross_entropy(action_preds, action_target)

            self._optimizer.zero_grad()
            action_loss.backward()
            torch.nn.utils.clip_grad_norm_(self._learn_model.parameters(), 0.25)
            self._optimizer.step()
            self._scheduler.step()

            self.log_action_losses.append(action_loss.detach().cpu().item())

        return {
            'cur_lr': self._optimizer.defaults['lr'],
            'action_loss': action_loss.item(),
        }

    def evaluate_on_env(self, state_mean=None, state_std=None, render=False):

        eval_batch_size = 1  # required for forward pass

        results = {}
        total_reward = 0
        total_timesteps = 0

        # state_dim = env.observation_space.shape[0]
        # act_dim = env.action_space.shape[0]

        if state_mean is None:
            self.state_mean = torch.zeros((self.state_dim, )).to(self.device)
        else:
            self.state_mean = torch.from_numpy(state_mean).to(self.device)

        if state_std is None:
            self.state_std = torch.ones((self.state_dim, )).to(self.device)
        else:
            self.state_std = torch.from_numpy(state_std).to(self.device)

        # same as timesteps used for training the transformer
        # also, crashes if device is passed to arange()
        timesteps = torch.arange(start=0, end=self.max_test_ep_len, step=1)
        timesteps = timesteps.repeat(eval_batch_size, 1).to(self.device)

        self._learn_model.eval()
        self._learn_model.to(self.device)  #TODO(pu)

        with torch.no_grad():

            for _ in range(self.num_eval_ep):

                # zeros place holders
                # continuous action
                actions = torch.zeros(
                    (eval_batch_size, self.max_test_ep_len, self.act_dim), dtype=torch.float32, device=self.device
                )

                # discrete action # TODO
                # actions = torch.randint(0,self.act_dim,[eval_batch_size, self.max_test_ep_len, 1], dtype=torch.long, device=self.device)

                states = torch.zeros(
                    (eval_batch_size, self.max_test_ep_len, self.state_dim), dtype=torch.float32, device=self.device
                )
                rewards_to_go = torch.zeros(
                    (eval_batch_size, self.max_test_ep_len, 1), dtype=torch.float32, device=self.device
                )

                # init episode
                running_state = self._env.reset()
                running_reward = 0
                running_rtg = self.rtg_target / self.rtg_scale

                for t in range(self.max_test_ep_len):

                    total_timesteps += 1

                    # add state in placeholder and normalize
                    states[0, t] = torch.from_numpy(running_state).to(self.device)
                    # states[0, t] = (states[0, t].cpu() - self.state_mean.cpu().numpy()) / self.state_std.cpu().numpy()
                    states[0, t] = (states[0, t] - self.state_mean) / self.state_std

                    # calcualate running rtg and add it in placeholder
                    running_rtg = running_rtg - (running_reward / self.rtg_scale)
                    rewards_to_go[0, t] = running_rtg

                    if t < self.context_len:
                        _, act_preds, _ = self._learn_model.forward(
                            timesteps[:, :self.context_len], states[:, :self.context_len],
                            actions[:, :self.context_len], rewards_to_go[:, :self.context_len]
                        )
                        act = act_preds[0, t].detach()
                    else:
                        _, act_preds, _ = self._learn_model.forward(
                            timesteps[:, t - self.context_len + 1:t + 1], states[:, t - self.context_len + 1:t + 1],
                            actions[:, t - self.context_len + 1:t + 1], rewards_to_go[:, t - self.context_len + 1:t + 1]
                        )
                        act = act_preds[0, -1].detach()

                    # if discrete
                    if not self._cfg.model.continuous:
                        act = torch.argmax(act)
                    running_state, running_reward, done, _ = self._env.step(act.cpu().numpy())

                    # add action in placeholder
                    actions[0, t] = act

                    total_reward += running_reward

                    if render:
                        self._env.render()
                    if done:
                        break

        results['eval/avg_reward'] = total_reward / self.num_eval_ep
        results['eval/avg_ep_len'] = total_timesteps / self.num_eval_ep

        return results

    def evaluate(self, state_mean=None, state_std=None, render=False):
        results = self.evaluate_on_env(state_mean, state_std, render)

        eval_avg_reward = results['eval/avg_reward']
        eval_avg_ep_len = results['eval/avg_ep_len']
        # eval_d4rl_score = self.get_d4rl_normalized_score(results['eval/avg_reward'], self.env_name) * 100

        mean_action_loss = np.mean(self.log_action_losses)
        time_elapsed = str(datetime.now().replace(microsecond=0) - self.start_time)

        self.total_updates += self.num_updates_per_iter

        log_str = (
            "=" * 60 + '\n' + "time elapsed: " + time_elapsed + '\n' + "num of updates: " + str(self.total_updates) + '\n' +
            "action loss: " + format(mean_action_loss, ".5f") + '\n' + "eval avg reward: " +
            format(eval_avg_reward, ".5f") + '\n' + "eval avg ep len: " + format(eval_avg_ep_len, ".5f")  #+ '\n' +
            # "eval d4rl score: " + format(eval_d4rl_score, ".5f")
        )

        print(log_str)

        # log_data = [time_elapsed, self.total_updates, mean_action_loss, eval_avg_reward, eval_avg_ep_len, eval_d4rl_score]
        log_data = [time_elapsed, self.total_updates, mean_action_loss, eval_avg_reward, eval_avg_ep_len]


        log_csv_name = self.prefix + "_log_" + self.start_time_str + ".csv"
        log_csv_path = os.path.join(self.log_dir, log_csv_name)

        csv_writer = csv.writer(open(log_csv_path, 'a', 1))
        # csv_header = (
        #     ["duration", "num_updates", "action_loss", "eval_avg_reward", "eval_avg_ep_len", "eval_d4rl_score"]
        # )
        csv_header = (
            ["duration", "num_updates", "action_loss", "eval_avg_reward", "eval_avg_ep_len"]
        )
        csv_writer.writerow(log_data)

        # save model
        # print("max d4rl score: " + format(max_d4rl_score, ".5f"))
        # if eval_d4rl_score >= max_d4rl_score:
        #     print("saving max d4rl score model at: " + self.save_best_model_path)
        #     torch.save(self._learn_model.state_dict(), self.save_best_model_path)
        #     max_d4rl_score = eval_d4rl_score

        # save model
        print("eval_avg_reward: " + format(eval_avg_reward, ".5f"))
        eval_cartpole_score = eval_avg_reward
        if  eval_cartpole_score >= self.max_cartpole_score:
            print("saving max cartpole score model at: " + self.save_best_model_path)
            torch.save(self._learn_model.state_dict(), self.save_best_model_path)
            self.max_cartpole_score = eval_cartpole_score

        print("saving current model at: " + self.save_model_path)
        torch.save(self._learn_model.state_dict(), self.save_model_path)
        
        stop=False
        if self.max_cartpole_score >= self.stop_value:
            stop=True
        return stop

    def get_d4rl_normalized_score(self, score, env_name):
        env_key = env_name.split('-')[0].lower()
        assert env_key in REF_MAX_SCORE, f'no reference score for {env_key} env to calculate d4rl score'
        return (score - REF_MIN_SCORE[env_key]) / (REF_MAX_SCORE[env_key] - REF_MIN_SCORE[env_key])

    def _state_dict_learn(self) -> Dict[str, Any]:
        return {
            'model': self._learn_model.state_dict(),
            # 'target_model': self._target_model.state_dict(),
            'optimizer': self._optimizer.state_dict(),
        }

    def _load_state_dict_learn(self, state_dict: Dict[str, Any]) -> None:
        self._learn_model.load_state_dict(state_dict['model'])
        # self._target_model.load_state_dict(state_dict['target_model'])
        self._optimizer.load_state_dict(state_dict['optimizer'])

    def default_model(self) -> Tuple[str, List[str]]:
        return 'dt', ['ding.model.template.decision_transformer']

    def _monitor_vars_learn(self) -> List[str]:
        return ['cur_lr', 'action_loss']
