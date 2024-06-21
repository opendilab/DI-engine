import copy
from collections import namedtuple
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F
import wandb

# from einops import pack, rearrange

from ding.model import model_wrap
from ding.torch_utils import Adam, to_device
from ding.utils import POLICY_REGISTRY
from ding.utils.data import default_collate, default_decollate

from .common_utils import default_preprocess_learn
from .sac import SACPolicy

QIntermediates = namedtuple(
    "QIntermediates", ["q_pred_all_actions", "q_pred", "q_next", "q_target"]
)


@POLICY_REGISTRY.register("qtransformer")
class QTransformerPolicy(SACPolicy):
    """
    Overview:
        Policy class of CQL algorithm for continuous control. Paper link: https://arxiv.org/abs/2006.04779.

    Config:
        == ====================  ========    =============  ================================= =======================
        ID Symbol                Type        Default Value  Description                       Other(Shape)
        == ====================  ========    =============  ================================= =======================
        1  ``type``              str                     | RL policy register name, refer  | this arg is optional,
                                                            | to registry ``POLICY_REGISTRY`` | a placeholder
        2  ``cuda``              bool        True           | Whether to use cuda for network |
        3  | ``random_``         int         10000          | Number of randomly collected    | Default to 10000 for
           | ``collect_size``                               | training samples in replay      | SAC, 25000 for DDPG/
           |                                                | buffer when training starts.    | TD3.
        4  | ``model.policy_``   int         256            | Linear layer size for policy    |
           | ``embedding_size``                             | network.                        |
        5  | ``model.soft_q_``   int         256            | Linear layer size for soft q    |
           | ``embedding_size``                             | network.                        |
        6  | ``model.value_``    int         256            | Linear layer size for value     | Defalut to None when
           | ``embedding_size``                             | network.                        | model.value_network
           |                                                |                                 | is False.
        7  | ``learn.learning``  float       3e-4           | Learning rate for soft q        | Defalut to 1e-3, when
           | ``_rate_q``                                    | network.                        | model.value_network
           |                                                |                                 | is True.
        8  | ``learn.learning``  float       3e-4           | Learning rate for policy        | Defalut to 1e-3, when
           | ``_rate_policy``                               | network.                        | model.value_network
           |                                                |                                 | is True.
        9  | ``learn.learning``  float       3e-4           | Learning rate for policy        | Defalut to None when
           | ``_rate_value``                                | network.                        | model.value_network
           |                                                |                                 | is False.
        10 | ``learn.alpha``     float       0.2            | Entropy regularization          | alpha is initiali-
           |                                                | coefficient.                    | zation for auto
           |                                                |                                 | `alpha`, when
           |                                                |                                 | auto_alpha is True
        11 | ``learn.repara_``   bool        True           | Determine whether to use        |
           | ``meterization``                               | reparameterization trick.       |
        12 | ``learn.``          bool        False          | Determine whether to use        | Temperature parameter
           | ``auto_alpha``                                 | auto temperature parameter      | determines the
           |                                                | `alpha`.                        | relative importance
           |                                                |                                 | of the entropy term
           |                                                |                                 | against the reward.
        13 | ``learn.-``         bool        False          | Determine whether to ignore     | Use ignore_done only
           | ``ignore_done``                                | done flag.                      | in halfcheetah env.
        14 | ``learn.-``         float       0.005          | Used for soft update of the     | aka. Interpolation
           | ``target_theta``                               | target network.                 | factor in polyak aver
           |                                                |                                 | aging for target
           |                                                |                                 | networks.
        == ====================  ========    =============  ================================= =======================
    """

    config = dict(
        # (str) RL policy register name (refer to function "POLICY_REGISTRY").
        type="qtransformer",
        # (bool) Whether to use cuda for policy.
        cuda=True,
        # (bool) on_policy: Determine whether on-policy or off-policy.
        # on-policy setting influences the behaviour of buffer.
        on_policy=False,
        # (bool) priority: Determine whether to use priority in buffer sample.
        priority=False,
        # (bool) Whether use Importance Sampling Weight to correct biased update. If True, priority must be True.
        priority_IS_weight=False,
        # (int) Number of training samples(randomly collected) in replay buffer when training starts.
        random_collect_size=10000,
        model=dict(
            # (bool type) twin_critic: Determine whether to use double-soft-q-net for target q computation.
            # Please refer to TD3 about Clipped Double-Q Learning trick, which learns two Q-functions instead of one .
            # Default to True.
            twin_critic=True,
            # (str type) action_space: Use reparameterization trick for continous action
            action_space="reparameterization",
            # (int) Hidden size for actor network head.
            actor_head_hidden_size=256,
            # (int) Hidden size for critic network head.
            critic_head_hidden_size=256,
        ),
        # learn_mode config
        learn=dict(
            # (int) How many updates (iterations) to train after collector's one collection.
            # Bigger "update_per_collect" means bigger off-policy.
            update_per_collect=1,
            # (int) Minibatch size for gradient descent.
            batch_size=256,
            # (float) learning_rate_q: Learning rate for soft q network.
            learning_rate_q=3e-4,
            # (float) learning_rate_policy: Learning rate for policy network.
            learning_rate_policy=3e-4,
            # (float) learning_rate_alpha: Learning rate for auto temperature parameter ``alpha``.
            learning_rate_alpha=3e-4,
            # (float) target_theta: Used for soft update of the target network,
            # aka. Interpolation factor in polyak averaging for target networks.
            target_theta=0.005,
            # (float) discount factor for the discounted sum of rewards, aka. gamma.
            discount_factor=0.99,
            # (float) alpha: Entropy regularization coefficient.
            # Please check out the original SAC paper (arXiv 1801.01290): Eq 1 for more details.
            # If auto_alpha is set  to `True`, alpha is initialization for auto `\alpha`.
            # Default to 0.2.
            alpha=0.2,
            # (bool) auto_alpha: Determine whether to use auto temperature parameter `\alpha` .
            # Temperature parameter determines the relative importance of the entropy term against the reward.
            # Please check out the original SAC paper (arXiv 1801.01290): Eq 1 for more details.
            # Default to False.
            # Note that: Using auto alpha needs to set learning_rate_alpha in `cfg.policy.learn`.
            auto_alpha=True,
            # (bool) log_space: Determine whether to use auto `\alpha` in log space.
            log_space=True,
            # (bool) Whether ignore done(usually for max step termination env. e.g. pendulum)
            # Note: Gym wraps the MuJoCo envs by default with TimeLimit environment wrappers.
            # These limit HalfCheetah, and several other MuJoCo envs, to max length of 1000.
            # However, interaction with HalfCheetah always gets done with done is False,
            # Since we inplace done==True with done==False to keep
            # TD-error accurate computation(``gamma * (1 - done) * next_v + reward``),
            # when the episode step is greater than max episode step.
            ignore_done=False,
            # (float) Weight uniform initialization range in the last output layer.
            init_w=3e-3,
            # (int) The numbers of action sample each at every state s from a uniform-at-random.
            num_actions=10,
            # (bool) Whether use lagrange multiplier in q value loss.
            with_lagrange=False,
            # (float) The threshold for difference in Q-values.
            lagrange_thresh=-1,
            # (float) Loss weight for conservative item.
            min_q_weight=1.0,
            # (bool) Whether to use entropy in target q.
            with_q_entropy=False,
        ),
        eval=dict(),  # for compatibility
    )

    def _init_learn(self) -> None:
        """
        Overview:
            Initialize the learn mode of policy, including related attributes and modules. For SAC, it mainly \
            contains three optimizers, algorithm-specific arguments such as gamma, min_q_weight, with_lagrange and \
            with_q_entropy, main and target model. Especially, the ``auto_alpha`` mechanism for balancing max entropy \
            target is also initialized here.
            This method will be called in ``__init__`` method if ``learn`` field is in ``enable_field``.

        .. note::
            For the member variables that need to be saved and loaded, please refer to the ``_state_dict_learn`` \
            and ``_load_state_dict_learn`` methods.

        .. note::
            For the member variables that need to be monitored, please refer to the ``_monitor_vars_learn`` method.

        .. note::
            If you want to set some spacial member variables in ``_init_learn`` method, you'd better name them \
            with prefix ``_learn_`` to avoid conflict with other modes, such as ``self._learn_attr1``.
        """
        self._priority = self._cfg.priority
        self._priority_IS_weight = self._cfg.priority_IS_weight
        self._num_actions = self._cfg.learn.num_actions

        self._min_q_version = 3
        self._min_q_weight = self._cfg.learn.min_q_weight
        self._with_lagrange = self._cfg.learn.with_lagrange and (
            self._lagrange_thresh > 0
        )
        self._lagrange_thresh = self._cfg.learn.lagrange_thresh
        if self._with_lagrange:
            self.target_action_gap = self._lagrange_thresh
            self.log_alpha_prime = torch.tensor(0.0).to(self._device).requires_grad_()
            self.alpha_prime_optimizer = Adam(
                [self.log_alpha_prime],
                lr=self._cfg.learn.learning_rate_q,
            )

        self._with_q_entropy = self._cfg.learn.with_q_entropy
        # Optimizers
        self._optimizer_q = Adam(
            self._model.parameters(),
            lr=self._cfg.learn.learning_rate_q,
        )

        # Algorithm config
        self._gamma = self._cfg.learn.discount_factor

        # Init auto alpha
        if self._cfg.learn.auto_alpha:
            if self._cfg.learn.target_entropy is None:
                assert (
                    "action_shape" in self._cfg.model
                ), "CQL need network model with action_shape variable"
                self._target_entropy = -np.prod(self._cfg.model.action_shape)
            else:
                self._target_entropy = self._cfg.learn.target_entropy
            if self._cfg.learn.log_space:
                self._log_alpha = torch.log(torch.FloatTensor([self._cfg.learn.alpha]))
                self._log_alpha = self._log_alpha.to(self._device).requires_grad_()
                self._alpha_optim = torch.optim.Adam(
                    [self._log_alpha], lr=self._cfg.learn.learning_rate_alpha
                )
                assert (
                    self._log_alpha.shape == torch.Size([1])
                    and self._log_alpha.requires_grad
                )
                self._alpha = self._log_alpha.detach().exp()
                self._auto_alpha = True
                self._log_space = True
            else:
                self._alpha = (
                    torch.FloatTensor([self._cfg.learn.alpha])
                    .to(self._device)
                    .requires_grad_()
                )
                self._alpha_optim = torch.optim.Adam(
                    [self._alpha], lr=self._cfg.learn.learning_rate_alpha
                )
                self._auto_alpha = True
                self._log_space = False
        else:
            self._alpha = torch.tensor(
                [self._cfg.learn.alpha],
                requires_grad=False,
                device=self._device,
                dtype=torch.float32,
            )
            self._auto_alpha = False
        for p in self._model.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)
        self._target_model = copy.deepcopy(self._model)
        self._target_model = model_wrap(
            self._target_model,
            wrapper_name="target",
            update_type="momentum",
            update_kwargs={"theta": self._cfg.learn.target_theta},
        )

        self._action_bin = self._cfg.model.action_bin

        self._action_values = np.array(
            [
                np.linspace(min_val, max_val, self._action_bin)
                for min_val, max_val in zip(
                    np.full(self._cfg.model.action_dim, -1),
                    np.full(self._cfg.model.action_dim, 1),
                )
            ]
        )
        # Main and target models
        self._learn_model = model_wrap(self._model, wrapper_name="base")
        self._learn_model.reset()
        self._target_model.reset()

        self._forward_learn_cnt = 0

    def _forward_learn(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Overview:
            Policy forward function of learn mode (training policy and updating parameters). Forward means \
            that the policy inputs some training batch data from the offline dataset and then returns the output \
            result, including various training information such as loss, action, priority.
        Arguments:
            - data (:obj:`List[Dict[int, Any]]`): The input data used for policy forward, including a batch of \
                training samples. For each element in list, the key of the dict is the name of data items and the \
                value is the corresponding data. Usually, the value is torch.Tensor or np.ndarray or there dict/list \
                combinations. In the ``_forward_learn`` method, data often need to first be stacked in the batch \
                dimension by some utility functions such as ``default_preprocess_learn``. \
                For CQL, each element in list is a dict containing at least the following keys: ``obs``, ``action``, \
                ``reward``, ``next_obs``, ``done``. Sometimes, it also contains other keys such as ``weight``.
        Returns:
            - info_dict (:obj:`Dict[str, Any]`): The information dict that indicated training result, which will be \
                recorded in text log and tensorboard, values must be python scalar or a list of scalars. For the \
                detailed definition of the dict, refer to the code of ``_monitor_vars_learn`` method.

        .. note::
            The input value can be torch.Tensor or dict/list combinations and current policy supports all of them. \
            For the data type that not supported, the main reason is that the corresponding model does not support it. \
            You can implement you own model rather than use the default model. For more information, please raise an \
            issue in GitHub repo and we will continue to follow up.
        """

        # data = default_preprocess_learn(
        #     data,
        #     use_priority=self._priority,
        #     use_priority_IS_weight=self._cfg.priority_IS_weight,
        #     ignore_done=self._cfg.learn.ignore_done,
        #     use_nstep=False,
        # )
        def discretization(x):
            self._action_values = torch.tensor(self._action_values)
            indices = torch.zeros_like(x, dtype=torch.long, device=x.device)
            for i in range(x.shape[1]):
                diff = (x[:, i].unsqueeze(-1) - self._action_values[i, :]) ** 2
                indices[:, i] = diff.argmin(dim=-1)
            return indices

        data["action"] = discretization(
            data["action"][:, -1, :]
        )  # torch.Size([2048, 10, 6]) -->torch.Size([2048, 6])
        data["next_action"] = discretization(
            data["next_action"][:, -1, :]
        )  # torch.Size([2048, 10, 6]) -->torch.Size([2048, 6])

        if self._cuda:
            data = to_device(data, self._device)

        self._learn_model.train()
        self._target_model.train()
        state = data["state"]  # torch.Size([2048, 10, 17])
        next_state = data["next_state"]  # torch.Size([2048, 10, 17])
        reward = data["reward"][:, -1]  # torch.Size([2048])
        done = data["done"][:, -1]  # torch.Size([2048])
        action = data["action"]  # torch.Size([2048, 6, 256])
        next_action = data["next_action"]  # torch.Size([2048, 6, 256])

        q_pred_all_actions = self._learn_model.forward(state, action=action)[:, 1:, :]
        # torch.Size([2048, 6, 256])

        def batch_select_indices(t, indices):
            indices = indices.unsqueeze(-1)
            selected = t.gather(-1, indices)
            selected = selected.squeeze(-1)
            return selected

        q_pred = batch_select_indices(q_pred_all_actions, action)
        # Create the dataset action mask and set selected values to 1
        dataset_action_mask = torch.zeros_like(q_pred_all_actions).scatter_(
            -1, action.unsqueeze(-1), 1
        )
        q_actions_not_taken = q_pred_all_actions[~dataset_action_mask.bool()]
        num_non_dataset_actions = q_actions_not_taken.size(0) // q_pred.size(0)
        conservative_loss = (
            (q_actions_not_taken - (0)) ** 2
        ).sum() / num_non_dataset_actions
        # Iterate over each row in the action tensor

        q_pred_rest_actions = q_pred[:, :-1]
        q_pred_last_action = q_pred[:, -1].unsqueeze(-1)
        with torch.no_grad():
            q_next_target = self._target_model.forward(next_state, action=next_action)[
                :, 1:, :
            ]
            q_target = self._target_model.forward(state, action=action)[:, 1:, :]

        q_target_rest_actions = q_target[:, 1:, :]
        max_q_target_rest_actions = q_target_rest_actions.max(dim=-1).values

        q_next_target_first_action = q_next_target[:, 0, :].unsqueeze(1)
        max_q_next_target_first_action = q_next_target_first_action.max(dim=-1).values

        losses_all_actions_but_last = F.mse_loss(
            q_pred_rest_actions, max_q_target_rest_actions
        )
        q_target_last_action = (reward * (1.0 - done.int())).unsqueeze(
            1
        ) + self._gamma * max_q_next_target_first_action
        losses_last_action = F.mse_loss(q_pred_last_action, q_target_last_action)
        td_loss = losses_all_actions_but_last + losses_last_action
        td_loss.mean()
        loss = td_loss + conservative_loss
        self._optimizer_q.zero_grad()
        loss.backward()
        self._optimizer_q.step()
        self._forward_learn_cnt += 1
        self._target_model.update(self._learn_model.state_dict())

        split_tensors = q_pred_all_actions.chunk(6, dim=1)
        q_means = [tensor.mean() for tensor in split_tensors]
        split_tensors_r = q_pred.chunk(6, dim=1)
        q_r_means = [tensor.mean() for tensor in split_tensors_r]
        wandb.log(
            {
                "td_loss": td_loss.item(),
                "losses_all_actions_but_last": losses_all_actions_but_last.item(),
                "losses_last_action": losses_last_action.item(),
                "conservative_loss": conservative_loss.item(),
                "q_mean": q_pred_all_actions.mean().item(),
                "q_a11": q_means[0].item(),
                "q_a12": q_means[1].item(),
                "q_a13": q_means[2].item(),
                "q_a14": q_means[3].item(),
                "q_a15": q_means[4].item(),
                "q_a16": q_means[5].item(),
                "q_r_a11": q_r_means[0].item(),
                "q_r_a12": q_r_means[1].item(),
                "q_r_a13": q_r_means[2].item(),
                "q_r_a14": q_r_means[3].item(),
                "q_r_a15": q_r_means[4].item(),
                "q_r_a16": q_r_means[5].item(),
                "q_all": q_pred_all_actions.mean().item(),
                "q_real": q_pred.mean().item(),
            },
        )
        return loss, q_pred_all_actions.mean().item()

    def _get_actions(self, obs):

        action = self._eval_model.get_actions(obs)
        action = 2.0 * action / (1.0 * self._action_bin) - 1.0
        return action

    # def _monitor_vars_learn(self) -> List[str]:
    #     """
    #     Overview:
    #         Return the necessary keys for logging the return dict of ``self._forward_learn``. The logger module, such \
    #         as text logger, tensorboard logger, will use these keys to save the corresponding data.
    #     Returns:
    #         - necessary_keys (:obj:`List[str]`): The list of the necessary keys to be logged.
    #     """
    #     return ["loss", "q_pred_all_actions.mean().item()"]

    def _state_dict_learn(self) -> Dict[str, Any]:
        """
        Overview:
            Return the state_dict of learn mode, usually including model, target_model and optimizers.
        Returns:
            - state_dict (:obj:`Dict[str, Any]`): The dict of current policy learn state, for saving and restoring.
        """
        ret = {
            "model": self._learn_model.state_dict(),
            "target_model": self._target_model.state_dict(),
            "optimizer_q": self._optimizer_q.state_dict(),
        }
        if self._auto_alpha:
            ret.update({"optimizer_alpha": self._alpha_optim.state_dict()})
        return ret

    def _load_state_dict_learn(self, state_dict: Dict[str, Any]) -> None:
        """
        Overview:
            Load the state_dict variable into policy learn mode.
        Arguments:
            - state_dict (:obj:`Dict[str, Any]`): The dict of policy learn state saved before.

        .. tip::
            If you want to only load some parts of model, you can simply set the ``strict`` argument in \
            load_state_dict to ``False``, or refer to ``ding.torch_utils.checkpoint_helper`` for more \
            complicated operation.
        """
        self._learn_model.load_state_dict(state_dict["model"])
        self._target_model.load_state_dict(state_dict["ema_model"])
        self._optimizer_q.load_state_dict(state_dict["optimizer_q"])
        if self._auto_alpha:
            self._alpha_optim.load_state_dict(state_dict["optimizer_alpha"])

    def _init_eval(self) -> None:
        self._eval_model = model_wrap(self._model, wrapper_name="base")
        self._eval_model.reset()

    def _forward_eval(self, data: dict) -> dict:
        r"""
        Overview:
            Forward function of eval mode, similar to ``self._forward_collect``.
        Arguments:
            - data (:obj:`Dict[str, Any]`): Dict type data, stacked env data for predicting policy_output(action), \
                values are torch.Tensor or np.ndarray or dict/list combinations, keys are env_id indicated by integer.
        Returns:
            - output (:obj:`Dict[int, Any]`): The dict of predicting action for the interaction with env.
        ReturnsKeys
            - necessary: ``action``
        """
        data_id = list(data.keys())
        data = default_collate(list(data.values()))
        if self._cuda:
            data = to_device(data, self._device)
        self._eval_model.eval()
        with torch.no_grad():
            output = self._get_actions(data)
        if self._cuda:
            output = to_device(output, "cpu")
        output = default_decollate(output)
        output = [{"action": o} for o in output]
        return {i: d for i, d in zip(data_id, output)}
