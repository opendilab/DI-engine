import os
from collections import namedtuple
from typing import List, Dict, Any, Tuple, Union

import torch.distributions
import torch.nn.functional as F

from ding.config.config import read_config_yaml
from ding.model import model_wrap
from ding.policy.base_policy import Policy
from ding.policy.alphazero.alphazero_helper import MCTS
from ding.torch_utils import Adam, to_device
from ding.utils import POLICY_REGISTRY
from ding.utils.data import default_collate


@POLICY_REGISTRY.register('alphazero')
class AlphaZeroPolicy(Policy):

    def _init_learn(self):
        # Optimizer
        self._grad_norm = self._cfg.learn.get('grad_norm', 1)
        self._learning_rate = self._cfg.learn.learning_rate
        self._weight_decay = self._cfg.learn.get('weight_decay', 0)
        self._optimizer = Adam(self._model.parameters(), weight_decay=self._weight_decay, lr=self._learning_rate)

        # Algorithm config
        self._value_weight = self._cfg.learn.get('value_weight', 1)
        # Main and target models
        self._learn_model = model_wrap(self._model, wrapper_name='base')
        self._learn_model.reset()

    def _forward_learn(self, inputs: dict) -> Dict[str, Any]:
        inputs = default_collate(inputs)
        if self._cuda:
            inputs = to_device(inputs, self._device)
        self._learn_model.train()

        state_batch = inputs['state']
        mcts_probs = inputs['mcts_prob']
        winner_batch = inputs['winner']

        state_batch = state_batch.to(device=self._device, dtype=torch.float)
        mcts_probs = mcts_probs.to(device=self._device, dtype=torch.float)
        winner_batch = winner_batch.to(device=self._device, dtype=torch.float)

        log_probs, values = self.compute_prob_value(state_batch)
        # ============
        # policy loss
        # ============
        policy_loss = -torch.mean(torch.sum(mcts_probs * log_probs))

        # ============
        # value loss
        # ============
        value_loss = F.mse_loss(values.view(-1), winner_batch)

        total_loss = self._value_weight * value_loss + policy_loss
        self._optimizer.zero_grad()
        total_loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            list(self._model.parameters()),
            max_norm=self._grad_norm,
        )
        self._optimizer.step()
        # calc policy entropy, for monitoring only
        entropy = -torch.mean(torch.sum(torch.exp(log_probs) * log_probs, 1))

        # =============
        # after update
        # =============
        return {
            'cur_lr': self._optimizer.param_groups[0]['lr'],
            'total_loss': total_loss,
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'entropy': entropy,
            'grad_norm': grad_norm,
        }

    def _init_collect(self) -> None:
        r"""
        Overview:
            Collect mode init method. Called by ``self.__init__``.
            Init traj and unroll length, collect model.
        """
        self._collect_mcts = MCTS(self._cfg.collect.mcts)
        self._collect_model = model_wrap(self._model, wrapper_name='base')
        self._collect_model.reset()

    @torch.no_grad()
    def _forward_collect(self, env):
        r"""
        Overview:
            Forward function for collect mode
        Arguments:
            - data (:obj:`dict`): Dict type data, including at least ['obs'].
        Returns:
            - data (:obj:`dict`): The collected data
        """
        action, move_probs = self._collect_mcts.get_next_action(
            env, policy_forward_fn=self._policy_value_fn, temperature=1.0, sample=True
        )
        return action, move_probs

    def _get_train_sample(self, data: list) -> Union[None, List[Any]]:
        pass

    def _process_transition(self, obs: Any, model_output: dict, timestep: namedtuple) -> dict:
        r"""
        Overview:
            Generate dict type transition data from inputs.
        Arguments:
            - obs (:obj:`Any`): Env observation
            - model_output (:obj:`dict`): Output of collect model, including at least ['action']
            - timestep (:obj:`namedtuple`): Output after env step, including at least ['obs', 'reward', 'done'] \
                (here 'obs' indicates obs after env step).
        Returns:
            - transition (:obj:`dict`): Dict type transition data.
        """
        transition = {
            'obs': obs,
            'next_obs': timestep.obs,
            'action': model_output['action'],
            'value': model_output['value'],
            'reward': timestep.reward,
            'done': timestep.done,
        }
        return transition

    def _init_eval(self) -> None:
        r"""
        Overview:
            Evaluate mode init method. Called by ``self.__init__``.
            Init eval model with argmax strategy.
        """
        self._eval_mcts = MCTS(self._cfg.eval.mcts)
        self._eval_model = model_wrap(self._model, wrapper_name='base')
        self._eval_model.reset()

    def _forward_eval(self, env: dict) -> dict:
        r"""
        Overview:
            Forward function for eval mode, similar to ``self._forward_collect``.
        Arguments:
            - data (:obj:`dict`): Dict type data, including at least ['obs'].
        Returns:
            - output (:obj:`dict`): Dict type data, including at least inferred action according to input obs.
        """
        action, move_probs = self._eval_mcts.get_next_action(
            env, policy_forward_fn=self._policy_value_fn, temperature=1.0, sample=False
        )
        return action, move_probs

    @torch.no_grad()
    def _policy_value_fn(self, env):
        """
        input: env
        output: a list of (action, probability) tuples for each available
        action and the score of the env state
        """
        legal_actions = env.legal_actions
        current_state = env.current_state()
        current_state = torch.from_numpy(current_state).to(device=self._device, dtype=torch.float).unsqueeze(0)
        with torch.no_grad():
            action_probs, value = self.compute_prob_value(current_state)
        # action_probs_zip = zip(legal_actions, action_probs.squeeze(0)[legal_actions].detach().numpy().tolist())
        action_probs_dict = dict(zip(legal_actions, action_probs.squeeze(0)[legal_actions].detach().numpy()))
        value = value.item()
        return action_probs_dict, value

    @staticmethod
    def set_learning_rate(optimizer, lr):
        """Sets the learning rate to the given value"""
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def compute_logit_value(self, state_batch):
        logits, values = self._model(state_batch)
        return logits, values

    def compute_prob_value(self, state_batch):
        logits, values = self._model(state_batch)
        dist = torch.distributions.Categorical(logits=logits)
        probs = dist.probs
        return probs, values

    def compute_logp_value(self, state_batch):
        logits, values = self._model(state_batch)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs, values

    def default_model(self) -> Tuple[str, List[str]]:
        return 'vac', ['ding.model.template.vac']

    def _monitor_vars_learn(self) -> List[str]:
        return super()._monitor_vars_learn() + ['policy_loss', 'value_loss', 'entropy_loss', 'grad_norm']


if __name__ == '__main__':
    cfg_path = os.path.join(os.getcwd(), 'alphazero_config_ding.yaml')
    cfg = read_config_yaml(cfg_path)
    policy = AlphaZeroPolicy(cfg)
