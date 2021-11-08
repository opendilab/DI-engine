from collections.abc import Iterable
from easydict import EasyDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ding.utils import REWARD_MODEL_REGISTRY
from .base_reward_model import BaseRewardModel
from ding.model.template.q_learning import DQN
from ding.model.template.vac import VAC
from torch.distributions.categorical import Categorical
import gym
import numpy as np
from dizoo.atari.envs.atari_wrappers import wrap_deepmind
from .rnd_reward_model import collect_states
from ding.utils import SequenceType
from ding.model.common import FCEncoder


class ConvEncoder(nn.Module):
    r"""
    Overview:
        The ``Convolution Encoder`` used in models. Used to encoder raw 2-dim observation.
    Interfaces:
        ``__init__``, ``forward``
    """

    def __init__(self, ) -> None:
        super(ConvEncoder, self).__init__()
        self.conv1 = nn.Conv2d(4, 16, 7, stride=3)
        self.conv2 = nn.Conv2d(16, 16, 5, stride=2)
        self.conv3 = nn.Conv2d(16, 16, 3, stride=1)
        self.conv4 = nn.Conv2d(16, 16, 3, stride=1)
        self.fc1 = nn.Linear(784, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 3, 1, 2)  # get into NCHW format
        # compute forward pass of reward network (we parallelize across frames so
        # batch size is length of partial trajectory)
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = F.leaky_relu(self.conv4(x))
        x = x.reshape(-1, 784)
        x = F.leaky_relu(self.fc1(x))
        r = self.fc2(x)
        return r


class TREX_Model(nn.Module):

    def __init__(self, obs_shape):
        super(TREX_Model, self).__init__()
        if isinstance(obs_shape, int) or len(obs_shape) == 1:
            self.encoder = FCEncoder(obs_shape, [512, 64])
        # Conv Encoder
        elif len(obs_shape) == 3:
            self.encoder = ConvEncoder()

    def cum_return(self, traj):
        '''calculate cumulative return of trajectory'''
        sum_rewards = 0
        sum_abs_rewards = 0
        r = self.encoder(traj)
        sum_rewards += torch.sum(r)
        sum_abs_rewards += torch.sum(torch.abs(r))
        return sum_rewards, sum_abs_rewards

    def forward(self, traj_i, traj_j):
        '''compute cumulative return for each trajectory and return logits'''
        cum_r_i, abs_r_i = self.cum_return(traj_i)
        cum_r_j, abs_r_j = self.cum_return(traj_j)
        return torch.cat((cum_r_i.unsqueeze(0), cum_r_j.unsqueeze(0)), 0), abs_r_i + abs_r_j


@REWARD_MODEL_REGISTRY.register('trex')
class TrexRewardModel(BaseRewardModel):
    """
    Overview:
        The Trex reward model class (https://arxiv.org/pdf/1904.06387.pdf)
    Interface:
        ``estimate``, ``train``, ``load_expert_data``, ``collect_data``, ``clear_date``, \
            ``__init__``, ``_train``,
    """
    config = dict(
        type='trex',
        learning_rate=1e-5,
        # expert_data_path='expert_data.pkl'
        update_per_collect=100,
        batch_size=64,
        # input_size=4,
        target_new_data_count=64,
        hidden_size=128,
    )

    def __init__(self, config: EasyDict, device: str, tb_logger: 'SummaryWriter') -> None:  # noqa
        """
        Overview:
            Initialize ``self.`` See ``help(type(self))`` for accurate signature.
        Arguments:
            - cfg (:obj:`EasyDict`): Training config
            - device (:obj:`str`): Device usage, i.e. "cpu" or "cuda"
            - tb_logger (:obj:`SummaryWriter`): Logger, defaultly set as 'SummaryWriter' for model summary
        """
        super(TrexRewardModel, self).__init__()
        self.cfg = config
        assert device in ["cpu", "cuda"] or "cuda" in device
        self.device = device
        self.tb_logger = tb_logger
        self.reward_model = TREX_Model(self.cfg.policy.model.get('obs_shape'))
        self.reward_model.to(self.device)
        self.pre_expert_data = []
        self.train_data = []
        self.expert_data_loader = None
        self.opt = optim.Adam(self.reward_model.parameters(), config.reward_model.learning_rate)
        self.train_iter = 0
        self.learning_returns = []
        self.learning_rewards = []
        self.training_obs = []
        self.training_labels = []
        self.num_trajs = 0  # number of downsampled full trajectories
        self.num_snippets = 6000  # number of short subtrajectories to sample
        self.min_snippet_length = 50  # minimum number of short subtrajectories to sample
        self.max_snippet_length = 100  # maximum number of short subtrajectories to sample
        self.l1_reg = 0
        self.load_expert_data()

    def generate_novice_demos(self):
        if hasattr(self.cfg.env, 'env_id'):
            env = wrap_deepmind(self.cfg.env.env_id)
        else:
            env = gym.make(self.cfg.reward_model.env_id)
        checkpoint_min = 150000
        checkpoint_max = 260000
        checkpoint_step = 10000
        checkpoints = []
        for i in range(checkpoint_min, checkpoint_max + checkpoint_step, checkpoint_step):
            checkpoints.append(str(i))
        print(checkpoints)
        for checkpoint in checkpoints:

            model_path = self.cfg.reward_model.expert_model_path + \
            '/ckpt/iteration_' + checkpoint + '.pth.tar'
            if self.cfg.reward_model.algo_for_model in {'dqn', 'sql'}:
                model = DQN(
                    obs_shape=self.cfg.policy.model.obs_shape,
                    action_shape=self.cfg.policy.model.action_shape,
                    encoder_hidden_size_list=self.cfg.policy.model.encoder_hidden_size_list,
                )
                model.load_state_dict(torch.load(model_path)['model'])
                episode_count = 1
                for i in range(episode_count):
                    done = False
                    traj = []
                    gt_rewards = []
                    r = 0
                    env.seed(self.cfg.seed + (int(checkpoint)-int(checkpoint_min))//int(checkpoint_step)) # in trex official implementation, they use the same initialisation.
                    ob = env.reset()
                    steps = 0
                    acc_reward = 0
                    while True:
                        obs_tensor = torch.tensor(ob).unsqueeze(0)
                        logit = model.forward(obs_tensor.float())['logit']
                        #dist = Categorical(logits=logit)
                        action = logit.argmax(dim=-1)
                        action = action.numpy()
                        if action.shape == (1, ):
                            action = action.squeeze()  # 0-dim array
                        ob, r, done, _ = env.step(action)
                        if isinstance(self.cfg.policy.model.get('obs_shape'), int) or len(
                                self.cfg.policy.model.get('obs_shape')) == 1:
                            ob_processed = ob
                        elif len(self.cfg.policy.model.get('obs_shape')) == 3:
                            ob_processed = torch.tensor(ob).permute(1, 2, 0).numpy()
                        traj.append(ob_processed)
                        gt_rewards.append(r)
                        steps += 1
                        acc_reward += r
                        if done:
                            print("checkpoint: {}, steps: {}, return: {}".format(checkpoint, steps, acc_reward))
                            break
                    print("traj length", len(traj))
                    print("demo length", len(self.pre_expert_data))
                    self.pre_expert_data.append(traj)
                    self.learning_returns.append(acc_reward)
                    self.learning_rewards.append(gt_rewards)
            elif self.cfg.reward_model.algo_for_model == 'ppo':
                if hasattr(self.cfg.env, 'env_id'):
                    model = VAC(        # Todo, do we have a better solution to have generality?
                        obs_shape=self.cfg.policy.model.get('obs_shape'),
                        action_shape=self.cfg.policy.model.get('action_shape'),
                        encoder_hidden_size_list=self.cfg.policy.model.get('encoder_hidden_size_list'),
                        actor_head_hidden_size=self.cfg.policy.model.get('actor_head_hidden_size'),
                        critic_head_hidden_size=self.cfg.policy.model.get('critic_head_hidden_size'),
                        critic_head_layer_num=self.cfg.policy.model.get('critic_head_layer_num'),
                    )   # self.cfg.policy.model
                else:
                    if self.cfg.reward_model.env_id == 'LunarLander-v2':
                        model = VAC(
                            obs_shape=self.cfg.policy.model.get('obs_shape'),
                            action_shape=self.cfg.policy.model.get('action_shape')
                        )
                    else:
                        model = VAC(        # Todo, do we have a better solution to have generality?
                            obs_shape=self.cfg.policy.model.get('obs_shape'),
                            action_shape=self.cfg.policy.model.get('action_shape'),
                            encoder_hidden_size_list=self.cfg.policy.model.get('encoder_hidden_size_list'),
                            actor_head_hidden_size=self.cfg.policy.model.get('actor_head_hidden_size'),
                            critic_head_hidden_size=self.cfg.policy.model.get('critic_head_hidden_size'),
                            critic_head_layer_num=self.cfg.policy.model.get('critic_head_layer_num'),
                        )
                model.load_state_dict(torch.load(model_path)['model'])
                episode_count = 1
                for i in range(episode_count):
                    done = False
                    traj = []
                    gt_rewards = []
                    r = 0
                    env.seed(self.cfg.seed + (int(checkpoint)-int(checkpoint_min))//int(checkpoint_step))
                    ob = env.reset()
                    steps = 0
                    acc_reward = 0
                    while True:
                        obs_tensor = torch.tensor(ob).unsqueeze(0)
                        logit = model.forward(obs_tensor.float(), mode='compute_actor_critic')['logit']
                        dist = Categorical(logits=logit)
                        #action = dist.argmax(dim=-1)
                        action = dist.sample()
                        action = action.numpy()
                        if action.shape == (1, ):
                            action = action.squeeze()  # 0-dim array
                        ob, r, done, _ = env.step(action)
                        if isinstance(self.cfg.policy.model.get('obs_shape'), int) or len(
                                self.cfg.policy.model.get('obs_shape')) == 1:
                            ob_processed = ob
                        elif len(self.cfg.policy.model.get('obs_shape')) == 3:
                            ob_processed = torch.tensor(ob).permute(1, 2, 0).numpy()
                        traj.append(ob_processed)
                        gt_rewards.append(r)
                        steps += 1
                        acc_reward += r
                        if done:
                            print("checkpoint: {}, steps: {}, return: {}".format(checkpoint, steps, acc_reward))
                            break
                    print("traj length", len(traj))
                    print("demo length", len(self.pre_expert_data))
                    self.pre_expert_data.append(traj)
                    self.learning_returns.append(acc_reward)
                    self.learning_rewards.append(gt_rewards)

        return self.pre_expert_data, self.learning_returns, self.learning_rewards

    def load_expert_data(self) -> None:
        """
        Overview:
            Getting the expert data from ``config.expert_data_path`` attribute in self
        Effects:
            This is a side effect function which updates the expert data attribute \
                (i.e. ``self.expert_data``) with ``fn:concat_state_action_pairs``
        """
        self.pre_expert_data, self.learning_returns, self.learning_rewards = self.generate_novice_demos()
        self.training_obs, self.training_labels = self.create_training_data()
        print("num_training_obs", len(self.training_obs))
        print("num_labels", len(self.training_labels))

    def create_training_data(self):
        demonstrations = self.pre_expert_data
        num_trajs = self.num_trajs
        num_snippets = self.num_snippets
        min_snippet_length = self.min_snippet_length
        max_snippet_length = self.max_snippet_length

        demo_lengths = [len(d) for d in demonstrations]
        print("demo_lengths", demo_lengths)
        max_snippet_length = min(np.min(demo_lengths), max_snippet_length)
        print("max snippet length", max_snippet_length)

        print(len(self.learning_returns))
        print(len(demonstrations))
        print([a[0] for a in zip(self.learning_returns, demonstrations)])
        demonstrations = [x for _, x in sorted(zip(self.learning_returns, demonstrations), key=lambda pair: pair[0])]
        sorted_returns = sorted(self.learning_returns)
        print(sorted_returns)

        #collect training data
        max_traj_length = 0
        num_demos = len(demonstrations)

        #add full trajs (for use on Enduro)
        for n in range(num_trajs):
            ti = 0
            tj = 0
            #only add trajectories that are different returns
            while (ti == tj):
                #pick two random demonstrations
                ti = np.random.randint(num_demos)
                tj = np.random.randint(num_demos)
            #create random partial trajs by finding random start frame and random skip frame
            si = np.random.randint(6)
            sj = np.random.randint(6)
            step = np.random.randint(3, 7)

            traj_i = demonstrations[ti][si::step]  # slice(start,stop,step)
            traj_j = demonstrations[tj][sj::step]

            if ti > tj:
                label = 0
            else:
                label = 1

            self.training_obs.append((traj_i, traj_j))
            self.training_labels.append(label)
            max_traj_length = max(max_traj_length, len(traj_i), len(traj_j))

        #fixed size snippets with progress prior
        for n in range(num_snippets):
            ti = 0
            tj = 0
            #only add trajectories that are different returns
            while (ti == tj):
                #pick two random demonstrations
                ti = np.random.randint(num_demos)
                tj = np.random.randint(num_demos)
            #create random snippets
            #find min length of both demos to ensure we can pick a demo no earlier
            #than that chosen in worse preferred demo
            min_length = min(len(demonstrations[ti]), len(demonstrations[tj]))
            rand_length = np.random.randint(min_snippet_length, max_snippet_length)
            if ti < tj:  # pick tj snippet to be later than ti
                ti_start = np.random.randint(min_length - rand_length + 1)
                # print(ti_start, len(demonstrations[tj]))
                tj_start = np.random.randint(ti_start, len(demonstrations[tj]) - rand_length + 1)
            else:  # ti is better so pick later snippet in ti
                tj_start = np.random.randint(min_length - rand_length + 1)
                # print(tj_start, len(demonstrations[ti]))
                ti_start = np.random.randint(tj_start, len(demonstrations[ti]) - rand_length + 1)
            traj_i = demonstrations[ti][ti_start:ti_start + rand_length:2]  # skip everyother framestack to reduce size
            traj_j = demonstrations[tj][tj_start:tj_start + rand_length:2]

            max_traj_length = max(max_traj_length, len(traj_i), len(traj_j))
            if ti > tj:
                label = 0
            else:
                label = 1
            self.training_obs.append((traj_i, traj_j))
            self.training_labels.append(label)

        print("maximum traj length", max_traj_length)
        return self.training_obs, self.training_labels

    def train(self):
        #check if gpu available
        device = self.device  # torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Assume that we are on a CUDA machine, then this should print a CUDA device:
        print(device)
        training_inputs, training_outputs = self.training_obs, self.training_labels
        loss_criterion = nn.CrossEntropyLoss()

        cum_loss = 0.0
        training_data = list(zip(training_inputs, training_outputs))
        for epoch in range(self.cfg.reward_model.update_per_collect):  # todo
            np.random.shuffle(training_data)
            training_obs, training_labels = zip(*training_data)
            for i in range(len(training_labels)):
                traj_i, traj_j = training_obs[i]
                labels = np.array([training_labels[i]])
                traj_i = np.array(traj_i)
                traj_j = np.array(traj_j)
                traj_i = torch.from_numpy(traj_i).float().to(device)
                traj_j = torch.from_numpy(traj_j).float().to(device)
                labels = torch.from_numpy(labels).to(device)

                #zero out gradient
                self.opt.zero_grad()

                #forward + backward + optimize
                outputs, abs_rewards = self.reward_model.forward(traj_i, traj_j)
                outputs = outputs.unsqueeze(0)
                loss = loss_criterion(outputs, labels) + self.l1_reg * abs_rewards
                loss.backward()
                self.opt.step()

                #print stats to see if learning
                item_loss = loss.item()
                cum_loss += item_loss
                if i % 100 == 99:
                    #print(i)
                    print("epoch {}:{} loss {}".format(epoch, i, cum_loss))
                    print(abs_rewards)
                    cum_loss = 0.0
                    print("check pointing")
                    torch.save(self.reward_model.state_dict(), self.cfg.reward_model.reward_model_path)
        torch.save(self.reward_model.state_dict(), self.cfg.reward_model.reward_model_path)
        print("finished training")
        # print out predicted cumulative returns and actual returns
        sorted_returns = sorted(self.learning_returns)
        with torch.no_grad():
            pred_returns = [self.predict_traj_return(self.reward_model, traj) for traj in self.pre_expert_data]
        for i, p in enumerate(pred_returns):
            print(i, p, sorted_returns[i])
        print("accuracy", self.calc_accuracy(self.reward_model, self.training_obs, self.training_labels))

    def predict_reward_sequence(self, net, traj):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        rewards_from_obs = []
        with torch.no_grad():
            for s in traj:
                r = net.cum_return(torch.from_numpy(np.array([s])).float().to(device))[0].item()
                rewards_from_obs.append(r)
        return rewards_from_obs

    def predict_traj_return(self, net, traj):
        return sum(self.predict_reward_sequence(net, traj))

    def calc_accuracy(self, reward_network, training_inputs, training_outputs):
        device = self.device
        loss_criterion = nn.CrossEntropyLoss()
        num_correct = 0.
        with torch.no_grad():
            for i in range(len(training_inputs)):
                label = training_outputs[i]
                traj_i, traj_j = training_inputs[i]
                traj_i = np.array(traj_i)
                traj_j = np.array(traj_j)
                traj_i = torch.from_numpy(traj_i).float().to(device)
                traj_j = torch.from_numpy(traj_j).float().to(device)

                #forward to get logits
                outputs, abs_return = reward_network.forward(traj_i, traj_j)
                _, pred_label = torch.max(outputs, 0)
                if pred_label.item() == label:
                    num_correct += 1.
        return num_correct / len(training_inputs)

    def estimate(self, data: list) -> None:
        """
        Overview:
            Estimate reward by rewriting the reward key in each row of the data.
        Arguments:
            - data (:obj:`list`): the list of data used for estimation, with at least \
                 ``obs`` and ``action`` keys.
        Effects:
            - This is a side effect function which updates the reward values in place.
        """
        res = collect_states(data)
        res = torch.stack(res).to(self.device)
        if isinstance(self.cfg.policy.model.get('obs_shape'), int) or len(self.cfg.policy.model.get('obs_shape')) == 1:
            pass
        elif len(self.cfg.policy.model.get('obs_shape')) == 3:
            res = res.permute(0, 3, 2, 1)
        reward = []
        with torch.no_grad():
            for i in range(res.shape[0]):
                sum_rewards, sum_abs_rewards = self.reward_model.cum_return(res)
                reward.append(sum_rewards)  # cpu?
        #sum_rewards = torch.chunk(reward, reward.shape[0], dim=0)
        sum_rewards = torch.stack(reward)
        for item, rew in zip(data, sum_rewards):
            item['reward'] = rew

    def collect_data(self, data: list) -> None:
        """
        Overview:
            Collecting training data formatted by  ``fn:concat_state_action_pairs``.
        Arguments:
            - data (:obj:`Any`): Raw training data (e.g. some form of states, actions, obs, etc)
        Effects:
            - This is a side effect function which updates the data attribute in ``self``
        """
        pass

    def clear_data(self) -> None:
        """
        Overview:
            Clearing training data. \
            This is a side effect function which clears the data attribute in ``self``
        """
        self.training_obs.clear()
        self.training_labels.clear()
