from typing import List, Dict, Union, Tuple, TYPE_CHECKING
import copy
import jax
import jax.numpy as jnp
import haiku as hk
from ding.jax_utils import q_1step_td_data, q_1step_td_error, AdamW, periodic_update, ArgmaxSampler, EpsGreedySampler
from .agent import Agent, NNState
if TYPE_CHECKING:
    from easydict import EasyDict


class DQNNetwork(hk.Module):

    def __init__(self, action_shape: int, hidden_size: List[int] = [64, 64], activation=jax.nn.relu) -> None:
        super(DQNNetwork, self).__init__()
        self.act = activation
        layers = []
        for h in hidden_size:
            layers.append(hk.Linear(h))
            layers.append(self.act)
        layers.append(hk.Linear(action_shape))
        self.main = hk.Sequential(layers)

    def __call__(self, x):
        return self.main(x)


class DQNAgent(Agent):
    config = dict(
        type='dqn_jax',
        cuda=False,
        on_policy=False,
        priority=False,
        priority_IS_weight=False,
        discount_factor=0.99,
        nstep=1,
        # learn
        update_per_collect=3,
        batch_size=64,
        learning_rate=1e-3,
        weight_decay=0.,
        target_update_freq=100,
        # collect
        random_collect_size=0,
        unroll_len=1,
        return_timestep_info=False,
        # other
        replay_buffer_size=int(1e4),
    )
    mode = ['learn', 'collect', 'eval']

    @classmethod
    def default_model(cls, rng_key, obs_space, action_space):
        dummy_data = jnp.ones([1, *obs_space.shape])
        return NNState(rng_key, dummy_data, create_model_fn=lambda: DQNNetwork(action_space.n))

    def __init__(self, cfg: "EasyDict", model: NNState, enable_mode: List[str] = None) -> None:
        self.cfg = cfg
        self.model = model
        if enable_mode is None:
            enable_mode = self.mode
        assert all([e in DQNAgent.mode for e in enable_mode]), enable_mode

        if 'learn' in enable_mode:
            self.optimizer = AdamW(self.model.param, lr=self.cfg.learning_rate, wd=self.cfg.weight_decay)
            self.target_model = copy.deepcopy(self.model)
        if 'collect' in enable_mode:
            self.collect_sampler = EpsGreedySampler()
        if 'eval' in enable_mode:
            self.eval_sampler = ArgmaxSampler()

    def learn(self, data) -> Dict:

        def dqn_loss(param, data):
            q_value = self.model.learn(param, data.obs)
            target_q_value = self.target_model.forward(data.next_obs)
            target_action = self.model.forward(data.next_obs).argmax(axis=-1)

            td_data = q_1step_td_data(q_value, target_q_value, data.action, target_action, data.reward, data.done)
            loss, td_error_per_sample = q_1step_td_error(td_data, gamma=self.cfg.discount_factor)
            info = {
                'priority': jnp.abs(td_error_per_sample),
                'total_loss': loss,
            }
            return loss, info

        info = Agent.apply_train(dqn_loss, self.model, self.optimizer, data)
        self.target_model.param = periodic_update(
            self.model.param, self.target_model.param, self.optimizer.step_count, self.cfg.target_update_freq
        )
        return info

    def collect(self, obs, eps: float):
        q_value = self.model.forward(obs)
        return self.collect_sampler(q_value, eps)

    def process_transition(self, obs: jnp.array, agent_output: Union[Dict, jnp.array], timestep: Tuple):
        next_obs, reward, done, info = timestep
        transition = {
            'obs': obs,
            'next_obs': obs,
            'action': agent_output,
            'reward': reward,
            'done': done,
        }
        if self.cfg.return_timestep_info:
            transition['info'] = info
        return transition

    def eval(self, obs):
        q_value = self.model.forward(obs)
        return self.eval_sampler(q_value)
