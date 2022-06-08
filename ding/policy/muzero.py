from typing import List, Dict, Any, Tuple, Union
import treetensor.torch as ttorch
import torch.optim as optim
from ding.model import model_wrap
from ding.utils import POLICY_REGISTRY
from .base_policy import Policy
import torch
import copy
import numpy as np
from torch.nn import L1Loss
import torch.nn.functional as F
from ding.rl_utils import get_nstep_return_data, get_train_sample
import ding.rl_utils.efficientzero.ctree.cytree as cytree
from ding.rl_utils.efficientzero.mcts import MCTS
from ding.rl_utils.efficientzero.utils import select_action

from ding.torch_utils import to_tensor, to_ndarray
# TODO(pu): atari or tictactoe
# from dizoo.board_games.atari.game_config.atari_config import game_config
from dizoo.board_games.tictactoe.config.tictactoe_config import game_config


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

    def _data_preprocess_learn(self, data: ttorch.Tensor):
        # TODO data augmentation before learning
        data = data.cuda(self.game_config.device)
        data = ttorch.stack(data)
        return data

    @staticmethod
    def _consist_loss_func(f1, f2):
        """Consistency loss function: similarity loss
        Parameters
        """
        f1 = F.normalize(f1, p=2., dim=-1, eps=1e-5)
        f2 = F.normalize(f2, p=2., dim=-1, eps=1e-5)
        return -(f1 * f2).sum(dim=1)

    @staticmethod
    def _get_max_entropy(action_shape: int) -> None:
        p = 1.0 / action_shape
        return -action_shape * p * np.log2(p)

    def _init_learn(self) -> None:
        self._metric_loss = torch.nn.L1Loss()
        self._cos = torch.nn.CosineSimilarity(dim=1, eps=1e-05)
        self._ce = ModifiedCrossEntropyLoss(reduction='none')
        self._optimizer = optim.SGD(
            self._model.parameters(),
            lr=self._cfg.learning_rate,
            momentum=self._cfg.momentum,
            weight_decay=self._cfg.weight_decay,
            # grad_clip_type=self._cfg.grad_clip_type,
            # clip_value=self._cfg.grad_clip_value,
        )
        # use model_wrapper for specialized demands of different modes
        self._target_model = copy.deepcopy(self._model)
        self._target_model = model_wrap(
            self._target_model,
            wrapper_name='target',
            update_type='assign',
            update_kwargs={'freq': self._cfg.learn.target_update_freq}
        )
        self._learn_model = model_wrap(self._model, wrapper_name='base')
        self._learn_model.reset()
        self._target_model.reset()
        self.game_config = game_config

    def _forward_learn(self, data: ttorch.Tensor) -> Dict[str, Union[float, int]]:
        # inputs_batch, targets_batch = data
        # TODO(pu): priority
        inputs_batch, targets_batch, replay_buffer = data

        obs_batch_ori, action_batch, mask_batch, indices, weights_lst, make_time = inputs_batch
        target_value_prefix, target_value, target_policy = targets_batch

        # [:, 0: config.stacked_observations * 3,:,:]
        # obs_batch_ori is the original observations in a batch
        # obs_batch is the observation for hat s_t (predicted hidden states from dynamics function)
        # obs_target_batch is the observations for s_t (hidden states from representation function)
        # to save GPU memory usage, obs_batch_ori contains (stack + unroll steps) frames

        # TODO(pu): / 255.0
        obs_batch_ori = torch.from_numpy(obs_batch_ori).to(game_config.device).float()
        obs_batch = obs_batch_ori[:, 0:game_config.stacked_observations * game_config.image_channel, :, :]
        obs_target_batch = obs_batch_ori[:, game_config.image_channel:, :, :]

        # do augmentations
        if game_config.use_augmentation:
            obs_batch = game_config.transform(obs_batch)
            obs_target_batch = game_config.transform(obs_target_batch)

        # use GPU tensor
        action_batch = torch.from_numpy(action_batch).to(game_config.device).unsqueeze(-1).long()
        mask_batch = torch.from_numpy(mask_batch).to(game_config.device).float()
        target_value_prefix = torch.from_numpy(target_value_prefix).to(game_config.device).float()
        target_value = torch.from_numpy(target_value).to(game_config.device).float()
        target_policy = torch.from_numpy(target_policy).to(game_config.device).float()
        weights = torch.from_numpy(weights_lst).to(game_config.device).float()

        batch_size = obs_batch.size(0)
        assert batch_size == game_config.batch_size == target_value_prefix.size(0)
        metric_loss = torch.nn.L1Loss()

        # some logs preparation
        other_log = {}
        other_dist = {}

        other_loss = {
            'l1': -1,
            'l1_1': -1,
            'l1_-1': -1,
            'l1_0': -1,
        }
        for i in range(game_config.num_unroll_steps):
            key = 'unroll_' + str(i + 1) + '_l1'
            other_loss[key] = -1
            other_loss[key + '_1'] = -1
            other_loss[key + '_-1'] = -1
            other_loss[key + '_0'] = -1

        # transform targets to categorical representation
        transformed_target_value_prefix = game_config.scalar_transform(target_value_prefix)
        target_value_prefix_phi = game_config.reward_phi(transformed_target_value_prefix)

        transformed_target_value = game_config.scalar_transform(target_value)
        target_value_phi = game_config.value_phi(transformed_target_value)
        value, _, policy_logits, hidden_state, reward_hidden = self._learn_model.initial_inference(obs_batch)
        # TODO(pu)
        value = to_tensor(value)
        policy_logits = to_tensor(policy_logits)
        hidden_state = to_tensor(hidden_state)
        reward_hidden = to_tensor(reward_hidden)

        scaled_value = game_config.inverse_value_transform(value)

        if game_config.vis_result:
            state_lst = hidden_state.detach().cpu().numpy()
            # state_lst = hidden_state

        predicted_value_prefixs = []
        # Note: Following line is just for logging.
        if game_config.vis_result:
            predicted_values, predicted_policies = scaled_value.detach().cpu(), torch.softmax(
                policy_logits, dim=1
            ).detach().cpu()

        # calculate the new priorities for each transition
        value_priority = L1Loss(reduction='none')(scaled_value.squeeze(-1), target_value[:, 0])
        value_priority = value_priority.data.cpu().numpy() + game_config.prioritized_replay_eps

        # loss of the first step
        value_loss = game_config.scalar_value_loss(value, target_value_phi[:, 0])
        policy_loss = -(torch.log_softmax(policy_logits, dim=1) * target_policy[:, 0]).sum(1)
        value_prefix_loss = torch.zeros(batch_size, device=game_config.device)
        consistency_loss = torch.zeros(batch_size, device=game_config.device)

        target_value_prefix_cpu = target_value_prefix.detach().cpu()
        gradient_scale = 1 / game_config.num_unroll_steps
        # loss of the unrolled steps
        for step_i in range(game_config.num_unroll_steps):
            # unroll with the dynamics function
            value, value_prefix, policy_logits, hidden_state, reward_hidden = self._learn_model.recurrent_inference(
                hidden_state, reward_hidden, action_batch[:, step_i]
            )

            # TODO(pu)
            value = to_tensor(value)
            value_prefix = to_tensor(value_prefix)
            policy_logits = to_tensor(policy_logits)
            hidden_state = to_tensor(hidden_state)
            reward_hidden = to_tensor(reward_hidden)

            beg_index = game_config.image_channel * step_i
            end_index = game_config.image_channel * (step_i + game_config.stacked_observations)

            # consistency loss
            if game_config.consistency_coeff > 0:
                # obtain the oracle hidden states from representation function
                _, _, _, presentation_state, _ = self._learn_model.initial_inference(
                    obs_target_batch[:, beg_index:end_index, :, :]
                )

                hidden_state = to_tensor(hidden_state)
                presentation_state = to_tensor(presentation_state)

                # no grad for the presentation_state branch
                dynamic_proj = self._learn_model.project(hidden_state, with_grad=True)
                observation_proj = self._learn_model.project(presentation_state, with_grad=False)
                temp_loss = self._consist_loss_func(dynamic_proj, observation_proj) * mask_batch[:, step_i]

                other_loss['consist_' + str(step_i + 1)] = temp_loss.mean().item()
                consistency_loss += temp_loss

            policy_loss += -(torch.log_softmax(policy_logits, dim=1) * target_policy[:, step_i + 1]).sum(1)
            value_loss += game_config.scalar_value_loss(value, target_value_phi[:, step_i + 1])
            value_prefix_loss += game_config.scalar_reward_loss(value_prefix, target_value_prefix_phi[:, step_i])

            # Follow MuZero, set half gradient
            # hidden_state.register_hook(lambda grad: grad * 0.5)

            # reset hidden states
            if (step_i + 1) % game_config.lstm_horizon_len == 0:
                reward_hidden = (
                    torch.zeros(1, game_config.batch_size, game_config.lstm_hidden_size).to(game_config.device),
                    torch.zeros(1, game_config.batch_size, game_config.lstm_hidden_size).to(game_config.device)
                )

            if game_config.vis_result:
                scaled_value_prefixs = game_config.inverse_reward_transform(value_prefix.detach())
                scaled_value_prefixs_cpu = scaled_value_prefixs.detach().cpu()

                predicted_values = torch.cat(
                    (predicted_values, game_config.inverse_value_transform(value).detach().cpu())
                )
                predicted_value_prefixs.append(scaled_value_prefixs_cpu)
                predicted_policies = torch.cat((predicted_policies, torch.softmax(policy_logits, dim=1).detach().cpu()))
                state_lst = np.concatenate((state_lst, hidden_state.detach().cpu().numpy()))

                key = 'unroll_' + str(step_i + 1) + '_l1'

                value_prefix_indices_0 = (target_value_prefix_cpu[:, step_i].unsqueeze(-1) == 0)
                value_prefix_indices_n1 = (target_value_prefix_cpu[:, step_i].unsqueeze(-1) == -1)
                value_prefix_indices_1 = (target_value_prefix_cpu[:, step_i].unsqueeze(-1) == 1)

                target_value_prefix_base = target_value_prefix_cpu[:, step_i].reshape(-1).unsqueeze(-1)

                other_loss[key] = metric_loss(scaled_value_prefixs_cpu, target_value_prefix_base)
                if value_prefix_indices_1.any():
                    other_loss[key + '_1'] = metric_loss(
                        scaled_value_prefixs_cpu[value_prefix_indices_1],
                        target_value_prefix_base[value_prefix_indices_1]
                    )
                if value_prefix_indices_n1.any():
                    other_loss[key + '_-1'] = metric_loss(
                        scaled_value_prefixs_cpu[value_prefix_indices_n1],
                        target_value_prefix_base[value_prefix_indices_n1]
                    )
                if value_prefix_indices_0.any():
                    other_loss[key + '_0'] = metric_loss(
                        scaled_value_prefixs_cpu[value_prefix_indices_0],
                        target_value_prefix_base[value_prefix_indices_0]
                    )
        # ----------------------------------------------------------------------------------
        # weighted loss with masks (some invalid states which are out of trajectory.)
        loss = (
            game_config.consistency_coeff * consistency_loss + game_config.policy_loss_coeff * policy_loss +
            game_config.value_loss_coeff * value_loss + game_config.reward_loss_coeff * value_prefix_loss
        )
        weighted_loss = (weights * loss).mean()

        # backward
        parameters = self._learn_model.parameters()

        total_loss = weighted_loss
        total_loss.register_hook(lambda grad: grad * gradient_scale)
        self._optimizer.zero_grad()

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, game_config.max_grad_norm)
        self._optimizer.step()

        # ----------------------------------------------------------------------------------
        # update priority
        new_priority = value_priority
        # replay_buffer.update_priorities.remote(indices, new_priority, make_time)
        replay_buffer.batch_update(indices=indices, metas={'make_time': make_time, 'batch_priorities': new_priority})

        # packing data for logging
        loss_data = (
            total_loss.item(), weighted_loss.item(), loss.mean().item(), 0, policy_loss.mean().item(),
            value_prefix_loss.mean().item(), value_loss.mean().item(), consistency_loss.mean()
        )
        if game_config.vis_result:
            reward_w_dist, representation_mean, dynamic_mean, reward_mean = self._learn_model.get_params_mean()
            other_dist['reward_weights_dist'] = reward_w_dist
            other_log['representation_weight'] = representation_mean
            other_log['dynamic_weight'] = dynamic_mean
            other_log['reward_weight'] = reward_mean

            # reward l1 loss
            value_prefix_indices_0 = (
                target_value_prefix_cpu[:, :game_config.num_unroll_steps].reshape(-1).unsqueeze(-1) == 0
            )
            value_prefix_indices_n1 = (
                target_value_prefix_cpu[:, :game_config.num_unroll_steps].reshape(-1).unsqueeze(-1) == -1
            )
            value_prefix_indices_1 = (
                target_value_prefix_cpu[:, :game_config.num_unroll_steps].reshape(-1).unsqueeze(-1) == 1
            )

            target_value_prefix_base = target_value_prefix_cpu[:, :game_config.
                                                               num_unroll_steps].reshape(-1).unsqueeze(-1)

            predicted_value_prefixs = torch.stack(predicted_value_prefixs).transpose(1, 0).squeeze(-1)
            predicted_value_prefixs = predicted_value_prefixs.reshape(-1).unsqueeze(-1)
            other_loss['l1'] = metric_loss(predicted_value_prefixs, target_value_prefix_base)
            if value_prefix_indices_1.any():
                other_loss['l1_1'] = metric_loss(
                    predicted_value_prefixs[value_prefix_indices_1], target_value_prefix_base[value_prefix_indices_1]
                )
            if value_prefix_indices_n1.any():
                other_loss['l1_-1'] = metric_loss(
                    predicted_value_prefixs[value_prefix_indices_n1], target_value_prefix_base[value_prefix_indices_n1]
                )
            if value_prefix_indices_0.any():
                other_loss['l1_0'] = metric_loss(
                    predicted_value_prefixs[value_prefix_indices_0], target_value_prefix_base[value_prefix_indices_0]
                )

            td_data = (
                new_priority, target_value_prefix.detach().cpu().numpy(), target_value.detach().cpu().numpy(),
                transformed_target_value_prefix.detach().cpu().numpy(), transformed_target_value.detach().cpu().numpy(),
                target_value_prefix_phi.detach().cpu().numpy(), target_value_phi.detach().cpu().numpy(),
                predicted_value_prefixs.detach().cpu().numpy(), predicted_values.detach().cpu().numpy(),
                target_policy.detach().cpu().numpy(), predicted_policies.detach().cpu().numpy(), state_lst, other_loss,
                other_log, other_dist
            )
            priority_data = (weights, indices)
        else:
            td_data, priority_data = None, None

        return {
            'total_loss': loss_data[0],
            'weighted_loss': loss_data[1],
            'loss_mean': loss_data[2],
            'policy_loss': loss_data[4],
            'value_prefix_loss': loss_data[5],
            'value_loss': loss_data[6],
            'consistency_loss': loss_data[7],
            # 'td_data': td_data,
            # 'priority_data_weights': priority_data[0],
            # 'priority_data_indices': priority_data[1]
        }

    def _init_collect(self) -> None:
        self._unroll_len = self._cfg.collect.unroll_len
        self._collect_model = model_wrap(self._model, 'base')
        self._mcts_eval = MCTS(game_config)

    def _forward_collect(self, data: ttorch.Tensor, action_mask: list = None):
        """
        Shapes:
            obs: (B, S, C, H, W), where S is the stack num
            temperature: (N1, ), where N1 is the number of collect_env.
        """
        # TODO priority
        stack_obs = data
        game_config.env_num = len(stack_obs)
        with torch.no_grad():
            network_output = self._collect_model.initial_inference(stack_obs.float())
            hidden_state_roots = network_output.hidden_state  # （2, 64, 6, 6）
            reward_hidden_roots = network_output.reward_hidden  # {tuple:2} (1,2,512)
            value_prefix_pool = network_output.value_prefix  # {list: 2}
            policy_logits_pool = network_output.policy_logits.tolist()  # {list: 2} {list:6}

            # for atari:
            # roots = cytree.Roots(game_config.env_num, self.game_config.action_space_size, game_config.num_simulations)
            # noises = [
            #     np.random.dirichlet([game_config.root_dirichlet_alpha] *self.game_config.action_space_size
            #                         ).astype(np.float32).tolist() for _ in range(game_config.env_num)
            # ]

            # for board games:
            # TODO: when action_num is a list
            # action_num = [int(i.sum()) for i in action_mask]
            action_num = int(action_mask[0].sum())
            roots = cytree.Roots(game_config.env_num, action_num, game_config.num_simulations)
            # difference between collect and eval
            noises = [
                np.random.dirichlet([game_config.root_dirichlet_alpha] * action_num).astype(np.float32).tolist()
                for _ in range(game_config.env_num)
            ]
            roots.prepare(game_config.root_exploration_fraction, noises, value_prefix_pool, policy_logits_pool)
            # do MCTS for a policy (argmax in testing)
            self._mcts_eval.search(roots, self._collect_model, hidden_state_roots, reward_hidden_roots)

            roots_distributions = roots.get_distributions()  # {list: 1}->{list:6}
            roots_values = roots.get_values()  # {list: 1}
            data_id = [i for i in range(game_config.env_num)]
            output = {i: None for i in data_id}
            for i in range(game_config.env_num):
                distributions, value = roots_distributions[i], roots_values[i]
                # select the argmax, not sampling
                action, _ = select_action(distributions, temperature=1, deterministic=True)
                # TODO(pu): transform to real action index
                action = np.where(action_mask[i] == 1.0)[0][action]
                # actions.append(action)
                output[i] = {'action': action, 'distributions': distributions, 'value': value}

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
        data = get_nstep_return_data(data, self._nstep, gamma=self._gamma)
        return get_train_sample(data, self._unroll_len)

    def _init_eval(self) -> None:
        r"""
        Overview:
            Evaluate mode init method. Called by ``self.__init__``, initialize eval_model.
        """
        self._eval_model = model_wrap(self._model, wrapper_name='base')
        self._eval_model.eval()
        self._eval_model.reset()
        self._mcts_eval = MCTS(game_config)

    def _forward_eval(self, data: ttorch.Tensor, action_mask: list):
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
        self._eval_model.training = False
        game_config.env_num = 1
        stack_obs = data
        with torch.no_grad():
            # stack_obs {Tensor:(2,12,96,96)}
            network_output = self._eval_model.initial_inference(stack_obs.float())
            hidden_state_roots = network_output.hidden_state  # （2, 64, 6, 6）
            reward_hidden_roots = network_output.reward_hidden  # {tuple:2} (1,2,512)
            value_prefix_pool = network_output.value_prefix  # {list: 2}
            policy_logits_pool = network_output.policy_logits.tolist()  # {list: 2} {list:6}

            # for atari:
            # roots = cytree.Roots(game_config.env_num, game_config.action_space_size, game_config.num_simulations)
            # for board games:
            # TODO: when action_num is a list
            # action_num = [int(i.sum()) for i in action_mask]
            action_num = int(action_mask[0].sum())
            roots = cytree.Roots(game_config.env_num, action_num, game_config.num_simulations)
            roots.prepare_no_noise(value_prefix_pool, policy_logits_pool)
            # do MCTS for a policy (argmax in testing)
            self._mcts_eval.search(roots, self._eval_model, hidden_state_roots, reward_hidden_roots)

            roots_distributions = roots.get_distributions()  # {list: 1}->{list:6}
            roots_values = roots.get_values()  # {list: 1}
            data_id = [i for i in range(game_config.env_num)]
            output = {i: None for i in data_id}
            for i in range(game_config.env_num):
                distributions, value = roots_distributions[i], roots_values[i]
                # select the argmax, not sampling
                action, _ = select_action(distributions, temperature=1, deterministic=True)
                # TODO(pu): transform to real action index
                action = np.where(action_mask[i] == 1.0)[0][action]
                # actions.append(action)
                output[i] = {'action': action, 'distributions': distributions, 'value': value}

        return output

    def _monitor_vars_learn(self) -> List[str]:
        return [
            'total_loss',
            'weighted_loss',
            'loss_mean',
            'policy_loss',
            'value_prefix_loss',
            'value_loss',
            'consistency_loss',
            # 'td_data': td_data,
            'priority_data_weights',
            'priority_data_indices'
        ]

    def _state_dict_learn(self) -> Dict[str, Any]:
        """
        Overview:
            Return the state_dict of learn mode, usually including model and optimizer.
        Returns:
            - state_dict (:obj:`Dict[str, Any]`): the dict of current policy learn state, for saving and restoring.
        """
        return {
            'model': self._learn_model.state_dict(),
            'target_model': self._target_model.state_dict(),
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
        self._target_model.load_state_dict(state_dict['target_model'])
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
        # TODO(pu): atari or tictactoe
        # return 'EfficientZeroNet-atari', ['ding.model.template.model_based.efficientzero_atari_model']
        return 'EfficientZeroNet-tictactoe', ['ding.model.template.model_based.efficientzero_tictactoe_model']
