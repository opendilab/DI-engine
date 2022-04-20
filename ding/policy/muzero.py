from typing import List, Dict, Any, Tuple, Union
import treetensor.torch as ttorch

# from ding.torch_utils import SGD
import torch.optim as optim
from ding.model import model_wrap
from ding.utils import POLICY_REGISTRY
from .base_policy import Policy
import torch
import numpy as np
from tqdm.auto import tqdm
from torch.cuda.amp import autocast as autocast

import ding.rl_utils.efficientzero.ctree.cytree as cytree
from ding.rl_utils.efficientzero.game import GameHistory
from dizoo.board_games.atari.config.atari_config import game_config
from ding.rl_utils.efficientzero.mcts import MCTS
from ding.rl_utils.efficientzero.utils import select_action, prepare_observation_lst

class ModifiedCrossEntropyLoss(torch.nn.Module):

    def __init__(self, reduction: str = 'none'):
        assert reduction == 'none', reduction
        self.reduction = reduction

    def forward(self, inputs, target):
        return -(torch.log_softmax(inputs, dim=-1) * target).sum(dim=-1)


@POLICY_REGISTRY.register('muzero')
class MuZeroPolicy(Policy):
    """
    Overview:
        MuZero
        EfficientZero
    """
    config = dict(
        type='muzero',
        cuda=False,
        device='cpu',
        on_policy=False,
        priority=True,
        priority_IS_weight=True,
        batch_size=256,
        discount_factor=0.997,
        learning_rate=1e-3,
        momentum=0.9,
        weight_decay=1e-4,
        grad_clip_type='clip_norm',
        grad_clip_value=5,
        policy_weight=1.0,
        value_weight=0.25,
        consistent_weight=1.0,
        value_prefix_weight=2.0,
        image_unroll_len=5,
        lstm_horizon_len=5,
        # collect
        # collect_env_num=8,
        # action_shape=6,
        simulation_num=50,
        root_dirichlet_alpha=0.3,
        root_exploration_fraction=0.25,
        value_delta_max=0.01,

        # learn_mode config
        learn=dict(
            multi_gpu=False,
            update_per_collect=10,
            batch_size=64,
            learning_rate=0.001,
            # Frequency of target network update.
            target_update_freq=100,
            # grad_clip_type='clip_norm',
            # grad_clip_value=0.5,

        ),
        # collect_mode config
        collect=dict(
            # You can use either "n_sample" or "n_episode" in collector.collect.
            # Get "n_sample" samples per collect.
            n_sample=64,
            # Cut trajectories into pieces with length "unroll_len".
            unroll_len=1,
        ),
        # command_mode config
        other=dict(
            # Epsilon greedy with decay.
            eps=dict(
                # Decay type. Support ['exp', 'linear'].
                type='exp',
                start=0.95,
                end=0.1,
                decay=50000,
            ),
            replay_buffer=dict(replay_buffer_size=100000, type='game')
        ),
    )

    def _init_learn(self) -> None:
        self._metric_loss = torch.nn.L1Loss()
        self._cos = torch.nn.CosineSimilarity(dim=1, eps=1e-05)
        self._ce = ModifiedCrossEntropyLoss(reduction='none')
        # self._optimizer = SGD(  # TODO
        #     self._model.parameters(),
        #     lr=self._cfg.learning_rate,
        #     momentum=self._cfg.momentum,
        #     weight_decay=self._cfg.weight_decay,
        #     grad_clip_type=self._cfg.grad_clip_type,
        #     clip_value=self._cfg.grad_clip_value
        # )
        self._optimizer = optim.SGD(
            self._model.parameters(),
            lr=self._cfg.learning_rate,
            momentum=self._cfg.momentum,
            weight_decay=self._cfg.weight_decay,
            # grad_clip_type=self._cfg.grad_clip_type,
            # clip_value=self._cfg.grad_clip_value,
        )

        self._learn_model = model_wrap(self._model, wrapper_name='base')
        self._learn_model.reset()

    def _data_preprocess_learn(self, data: ttorch.Tensor):
        # TODO data augmentation before learning
        data = data.cuda(self.device)
        data = ttorch.stack(data)
        return data

    def _forward_learn(self, data: ttorch.Tensor) -> Dict[str, Union[float, int]]:
        self._learn_model.train()
        losses = ttorch.as_tensor({})
        losses.consistent_loss = torch.zeros(1).to(self.device)
        losses.value_prefix_loss = torch.zeros(1).to(self.device)

        # first step
        output = self._learn_model.forward(data.obs, mode='init')
        losses.value_loss = self._ce(output.value, data.target_value[0])
        td_error_per_sample = losses.value_loss.clone().detach()
        losses.policy_loss = self._ce(output.logit, data.target_action[0])

        # unroll N step
        N = self._cfg.image_unroll_len
        for i in range(N):
            if self._cfg.value_prefix_weight > 0:
                output = self._learn_model.forward(
                    output.hidden_state, output.hidden_state_reward, data.action[i], mode='recurrent'
                )
            else:
                output = self._learn_model.forward(output.hidden_state, data.action[i], mode='recurrent')
            losses.value_loss += self._ce(output.value, data.target_value[i + 1])
            losses.policy_loss += self._ce(output.logit, data.target_action[i + 1])
            # consistent loss
            if self._cfg.consistent_weight > 0:
                with torch.no_grad():
                    next_hidden_state = self._learn_model.forward(data.next_obs, mode='init').hidden_state
                    projected_next = self._learn_model.forward(next_hidden_state, mode='project')
                projected_now = self._learn_model.forward(output.hidden_state, mode='project')
                losses.consistent_loss += -(self._cos(projected_now, projected_next) * data.mask[i])
            # value prefix loss
            if self._cfg.value_prefix_weight > 0:
                losses.value_prefix_loss += self._ce(output.value_prefix, data.target_value_prefix[i])
            # set half gradient
            output.hidden_state.register_hook(lambda grad: grad * 0.5)
            # reset hidden states
            if (i + 1) % self._cfg.lstm_horizon_len == 0:
                output.hidden_state_reward.zero_()

        total_loss = (
                self._cfg.policy_weight * losses.policy_loss + self._cfg.value_weight * losses.value_loss +
                self._cfg.value_prefix_weight * losses.value_prefix_loss +
                self._cfg.consistent_weight * losses.consistent_loss
        )
        total_loss = total_loss.mean()
        total_loss.register_hook(lambda grad: grad / N)

        self._optimizer.zero_grad()
        total_loss.backward()
        self._optimizer.step()

        return {
            'total_loss': total_loss.item(),
            'priority': td_error_per_sample.abs().tolist(),
        }.update({k: v.mean().item()
                  for k, v in losses.items()})

    def _init_collect(self) -> None:
        self._unroll_len = self._cfg.collect.unroll_len
        self._action_shape = (6,)
        self._gamma = self._cfg.discount_factor  # necessary for parallel
        # self._nstep = self._cfg.nstep  # necessary for parallel
        self._collect_model = model_wrap(self._model, 'base')
        # self._mcts_handler = MCTS(
        #     discount=self._cfg.discount_factor,
        #     value_delta_max=self._cfg.value_delta_max,
        #     horizons=self._cfg.lstm_horizon_len,
        #     simulation_num=self._cfg.simulation_num
        # )
        self.config = self._cfg
        self._mcts_handler = MCTS(self.config)
        # self._reset_collect()

    @staticmethod
    def _get_max_entropy(action_shape: int) -> None:
        p = 1.0 / action_shape
        return -action_shape * p * np.log2(p)

    def _forward_collect(self, data: ttorch.Tensor, temperature: torch.Tensor):
        """
        Shapes:
            obs: (B, S, C, H, W), where S is the stack num
            temperature: (N1, ), where N1 is the number of collect_env.
        """
        assert len(data.obs.shape) == 5
        env_id = data.env_id
        self._collect_model.eval()
        # TODO priority

        with torch.no_grad():
            obs = data.obs / 255.  # TODO move it into env
            obs = obs.view(obs.shape[0], -1, *obs.shape[2:])
            output = self._collect_model.forward(obs, mode='init')

            # root = Root(root_num=len(env_id), action_num=self._cfg.action_shape, tree_nodes=self._cfg.simulation_num)
            root = cytree.Roots(root_num=len(env_id), action_num=self._cfg.action_shape,
                                tree_nodes=self._cfg.simulation_num)
            noise = np.random.dirichlet(self._cfg.root_dirichlet_alpha, size=(len(env_id), self._cfg.action_shape))
            root.prepare(
                self._cfg.root_exploration_fraction, noise,
                output.value_prefix.cpu().numpy(),
                output.logit.cpu().numpy()
            )
            self._mcts_handler.search(
                root, self._collect_model,
                output.hidden_state.cpu().numpy(),
                output.hidden_state_reward.cpu().numpy()
            )

            output.distribution = ttorch.as_tensor(root.get_distributions())  # TODO whether to device
            output.value = ttorch.as_tensor(root.get_values())
            distribution = output.distribution ** (1 / temperature)
            action_prob = distribution / distribution.sum(dim=-1)
            output.action = torch.multinomial(action_prob, dim=-1).squeeze(-1)
        return output

    def _process_transition(
            self, obs: ttorch.Tensor, policy_output: ttorch.Tensor, timestep: ttorch.Tensor
    ) -> ttorch.Tensor:
        return ttorch.as_tensor(
            {
                'obs': obs,
                'action': policy_output.action,
                'distribution': policy_output.distribution,
                'value': policy_output.value,
                'next_obs': timestep.obs,
                'reward': timestep.reward,
                'done': timestep.done,
            }
        )

    def _get_train_sample(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        pass

    def _init_eval(self) -> None:
        r"""
        Overview:
            Evaluate mode init method. Called by ``self.__init__``, initialize eval_model.
        """
        self._eval_model = model_wrap(self._model, wrapper_name='base')
        self._eval_model.eval()
        self._eval_model.reset()
        # self.config.device='cpu'
        self.config = self._cfg
        self._mcts_eval = MCTS(self.config)

    def _forward_eval(self, data: ttorch.Tensor, temperature: torch.Tensor = torch.tensor(1)):
        """
        Overview:
            Forward computation graph of eval mode(evaluate policy performance), at most cases, it is similar to \
            ``self._forward_collect``.
        Arguments:
            - data (:obj:`Dict[str, Any]`): Dict type data, stacked env data for predicting policy_output(action), \
                values are torch.Tensor or np.ndarray or dict/list combinations, keys are env_id indicated by integer.
        Returns:
            - output (:obj:`Dict[int, Any]`): The dict of predicting action for the interaction with env.
        ArgumentsKeys:
            - necessary: ``obs``
        ReturnsKeys
            - necessary: ``action``
        """
        self._eval_model.training = False  # TODO
        config = game_config
        test_episodes = 2

        stack_obs = data
        with autocast():
            # stack_obs {Tensor:(2,12,96,96)}
            network_output = self._eval_model.initial_inference(stack_obs.float())
        hidden_state_roots = network_output.hidden_state  # （2, 64, 6, 6）
        reward_hidden_roots = network_output.reward_hidden  # {tuple:2} (1,2,512)
        value_prefix_pool = network_output.value_prefix  # {list: 2}
        policy_logits_pool = network_output.policy_logits.tolist()  # {list: 2} {list:6}

        roots = cytree.Roots(test_episodes, config.action_space_size, config.num_simulations)
        roots.prepare_no_noise(value_prefix_pool, policy_logits_pool)
        # do MCTS for a policy (argmax in testing)
        self._mcts_eval.search(roots, self._eval_model, hidden_state_roots, reward_hidden_roots)

        roots_distributions = roots.get_distributions()  # {list: 1}->{list:6}
        roots_values = roots.get_values()  # {list: 1}
        data_id = [i for i in range(test_episodes)]
        output = {i: None for i in data_id}
        for i in range(test_episodes):
            # if dones[i]:
            #     continue

            distributions, value = roots_distributions[i], roots_values[i]
            # select the argmax, not sampling
            action, _ = select_action(distributions, temperature=1, deterministic=True)

            # actions.append(action)
            output[i] = {'action': action, 'distributions': distributions,'value':value}

        # return {i: d for i, d in zip(data_id, output)}
        return output
        # return output

    def eval(self):
        """
        Overview:
            self-consistent eval method for EfficientZero
        """
        config = game_config

        exp_path = './'  # TODO
        render = False
        # render = True

        save_video = False
        final_test = False
        use_pb = True
        counter = 0
        config.device = 'cpu'
        device = config.device
        test_episodes = 3
        config.max_moves = 20
        config.env_name = 'PongNoFrameskip-v4'
        config.obs_shape = (12, 96, 96)
        config.gray_scale = False
        config.action_space_size = 6
        config.amp_type = 'none'
        # to obtain model = EfficientZeroNet()
        # model = config.get_uniform_network()
        model = self._eval_model
        model.to(device)
        model.eval()
        # save_path = os.path.join(config.exp_path, 'recordings', 'step_{}'.format(counter))

        if use_pb:
            pb = tqdm(np.arange(config.max_moves), leave=True)

        with torch.no_grad():
            # new games
            envs = [config.new_game(seed=i, save_video=save_video, save_path=None, test=True, final_test=final_test,
                                    video_callable=lambda episode_id: True, uid=i) for i in range(test_episodes)]

            # initializations
            init_obses = [env.reset() for env in envs]
            dones = np.array([False for _ in range(test_episodes)])
            game_histories = [
                GameHistory(envs[_].env.action_space, max_length=config.max_moves, config=config) for
                _ in
                range(test_episodes)]
            for i in range(test_episodes):
                game_histories[i].init([init_obses[i] for _ in range(config.stacked_observations)])

            step = 0
            ep_ori_rewards = np.zeros(test_episodes)
            ep_clip_rewards = np.zeros(test_episodes)
            # loop
            while not dones.all():
                if render:
                    for i in range(test_episodes):
                        envs[i].render()

                if config.image_based:
                    stack_obs = []
                    for game_history in game_histories:
                        stack_obs.append(game_history.step_obs())
                    stack_obs = prepare_observation_lst(stack_obs)
                    stack_obs = torch.from_numpy(stack_obs).to(device).float() / 255.0
                else:
                    stack_obs = [game_history.step_obs() for game_history in game_histories]
                    stack_obs = torch.from_numpy(np.array(stack_obs)).to(device)

                with autocast():
                    network_output = model.initial_inference(stack_obs.float())
                hidden_state_roots = network_output.hidden_state  # （1, 64, 6, 6）
                reward_hidden_roots = network_output.reward_hidden  # {tuple:2} (1,1,512)
                value_prefix_pool = network_output.value_prefix  # {list: 1}
                policy_logits_pool = network_output.policy_logits.tolist()  # {list: 1} {list:6}

                roots = cytree.Roots(test_episodes, config.action_space_size, config.num_simulations)
                roots.prepare_no_noise(value_prefix_pool, policy_logits_pool)
                # do MCTS for a policy (argmax in testing)
                MCTS(config).search(roots, model, hidden_state_roots, reward_hidden_roots)

                roots_distributions = roots.get_distributions()  # {list: 1}->{list:6}
                roots_values = roots.get_values()  # {list: 1}
                for i in range(test_episodes):
                    if dones[i]:
                        continue

                    distributions, value, env = roots_distributions[i], roots_values[i], envs[i]
                    # select the argmax, not sampling
                    action, _ = select_action(distributions, temperature=1, deterministic=True)

                    obs, ori_reward, done, info = env.step(action)
                    if config.clip_reward:
                        clip_reward = np.sign(ori_reward)
                    else:
                        clip_reward = ori_reward

                    game_histories[i].store_search_stats(distributions, value)
                    game_histories[i].append(action, obs, clip_reward)

                    dones[i] = done
                    ep_ori_rewards[i] += ori_reward
                    ep_clip_rewards[i] += clip_reward

                step += 1
                if use_pb:
                    pb.set_description('{} In step {}, scores: {}(max: {}, min: {}) currently.'
                                       ''.format(config.env_name, counter,
                                                 ep_ori_rewards.mean(), ep_ori_rewards.max(), ep_ori_rewards.min()))
                    pb.update(1)

            for env in envs:
                env.close()

    def _monitor_vars_learn(self) -> List[str]:
        return ['cur_lr', 'total_loss', 'q_value']

    def _state_dict_learn(self) -> Dict[str, Any]:
        """
        Overview:
            Return the state_dict of learn mode, usually including model and optimizer.
        Returns:
            - state_dict (:obj:`Dict[str, Any]`): the dict of current policy learn state, for saving and restoring.
        """
        return {
            'model': self._learn_model.state_dict(),
            # 'target_model': self._target_model.state_dict(),
            'optimizer': self._optimizer.state_dict(),
        }

    def _load_state_dict_learn(self, state_dict: Dict[str, Any]) -> None:
        """
        Overview:
            Load the state_dict variable into policy learn mode.
        Arguments:
            - state_dict (:obj:`Dict[str, Any]`): the dict of policy learn state saved before.

        .. tip::
            If you want to only load some parts of model, you can simply set the ``strict`` argument in \
            load_state_dict to ``False``, or refer to ``ding.torch_utils.checkpoint_helper`` for more \
            complicated operation.
        """
        self._learn_model.load_state_dict(state_dict['model'])
        # self._target_model.load_state_dict(state_dict['target_model'])
        self._optimizer.load_state_dict(state_dict['optimizer'])

    def default_model(self) -> Tuple[str, List[str]]:
        """
        Overview:
            Return this algorithm default model setting for demonstration.
        Returns:
            - model_info (:obj:`Tuple[str, List[str]]`): model name and mode import_names

        .. note::
            The user can define and use customized network model but must obey the same inferface definition indicated \
            by import_names path. For DQN, ``ding.model.template.q_learning.DQN``
        """
        return 'EfficientZeroNet-atari', ['ding.model.template.model_based.efficientzero_atari_model']
