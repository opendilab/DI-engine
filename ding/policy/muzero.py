from typing import List, Dict, Any, Tuple, Union
import numpy as np
import torch
import treetensor.torch as ttorch

# from ding.torch_utils import SGD
import torch.optim as optim
from ding.model import model_wrap
from ding.utils import POLICY_REGISTRY
# from ding.rl_utils.efficientzero_mcts.efficientzero_mcts import MCTS, Root
from ding.rl_utils.efficientzero_mcts.mcts import MCTS
import ding.rl_utils.efficientzero_mcts.ctree.cytree as cytree
from .base_policy import Policy


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
            grad_clip_type=self._cfg.grad_clip_type,
            clip_value=self._cfg.grad_clip_value
        )

        self._learn_model = model_wrap(self._model, wrapper='base')
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
        self._collect_model = model_wrap(self._model, 'base')
        self._mcts_handler = MCTS(
            discount=self._cfg.discount_factor,
            value_delta_max=self._cfg.value_delta_max,
            horizons=self._cfg.lstm_horizon_len,
            simulation_num=self._cfg.simulation_num
        )

        self._reset_collect()

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
        """
        Overview:
            For a given trajectory(transitions, a list of transition) data, process it into a list of sample that \
            can be used for training directly. A train sample can be a processed transition(DQN with nstep TD) \
            or some continuous transitions(DRQN).
        Arguments:
            - data (:obj:`List[Dict[str, Any]`): The trajectory data(a list of transition), each element is the same \
                format as the return value of ``self._process_transition`` method.
        Returns:
            - samples (:obj:`dict`): The list of training samples.

        .. note::
            We will vectorize ``process_transition`` and ``get_train_sample`` method in the following release version. \
            And the user can customize the this data processing procecure by overriding this two methods and collector \
            itself.
        """
        data = get_nstep_return_data(data, self._nstep, gamma=self._gamma)
        return get_train_sample(data, self._unroll_len)

    def _reset_collect(self, env_id: List[int] = None):
        self._collect_model.reset(env_id=env_id)

    def _init_eval(self) -> None:
        r"""
        Overview:
            Evaluate mode init method. Called by ``self.__init__``, initialize eval_model.
        """
        self._eval_model = model_wrap(self._model, wrapper_name='base')
        self._eval_model.reset()

    def _forward_eval(self, data: Dict[int, Any]) -> Dict[int, Any]:
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
        data_id = list(data.keys())
        data = default_collate(list(data.values()))
        if self._cuda:
            data = to_device(data, self._device)
        self._eval_model.eval()
        with torch.no_grad():
            output = self._eval_model.forward(data)
        if self._cuda:
            output = to_device(output, 'cpu')
        output = default_decollate(output)
        return {i: d for i, d in zip(data_id, output)}

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
        return 'EfficientZeroNet', ['ding.model.template.model_based.efficientzero_tictactoe_model']
