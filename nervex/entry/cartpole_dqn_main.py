import time
import copy
import argparse
import torch
import os

from nervex.envs.gym.cartpole.cartpole_env import CartpoleEnv
from nervex.worker import BaseLearner, SubprocessEnvManager
from nervex.worker.agent.sumo_dqn_agent import SumoDqnActorAgent
from nervex.utils import read_config
from nervex.entry.base_single_machine import SingleMachineRunner
from nervex.worker.agent import BaseAgent, IAgentStatelessPlugin
from collections import OrderedDict
from nervex.computation_graph import BaseCompGraph
from nervex.model import FCDQN
from nervex.rl_utils import td_data, one_step_td_error


class CartpoleDqnGraph(BaseCompGraph):
    def __init__(self, cfg):
        self._gamma = cfg.dqn.discount_factor

    def forward(self, data, agent):
        obs = data.get('obs')
        nextobs = data.get('next_obs')
        reward = data.get('reward').squeeze(1)
        action = data.get('action')
        terminate = data.get('done').float()
        weights = data.get('weights', None)

        q_value = agent.forward(obs)
        if agent.is_double:
            target_q_value = agent.target_forward(nextobs)
        else:
            target_q_value = agent.forward(nextobs)

        data = td_data(q_value, target_q_value, action, reward, terminate)
        loss = one_step_td_error(data, self._gamma, weights)
        if agent.is_double:
            agent.update_target_network(agent.state_dict()['model'])
        return {'total_loss': loss}

    def __repr__(self):
        return "CartpoleDqnGraph"

    def register_stats(self, recorder, tb_logger):
        recorder.register_var('total_loss')
        tb_logger.register_var('total_loss')

    def get_weighted_reward(self, reward):
        return reward


class CartpoleDqnLearnerAgent(BaseAgent):
    def __init__(self, model, is_double=True):
        self.plugin_cfg = OrderedDict({
            'grad': {
                'enable_grad': True
            },
        })
        # whether use double(target) q-network plugin
        if is_double:
            # self.plugin_cfg['target_network'] = {'update_cfg': {'type': 'momentum', 'kwargs': {'theta': 0.001}}}
            self.plugin_cfg['target_network'] = {'update_cfg': {'type': 'assign', 'kwargs': {'freq': 500}}}
        self.is_double = is_double
        super(CartpoleDqnLearnerAgent, self).__init__(model, self.plugin_cfg)


class CartpoleDqnActorAgent(BaseAgent):
    def __init__(self, model):
        plugin_cfg = OrderedDict({
            'eps_greedy_sample': {},
            'grad': {
                'enable_grad': False
            },
        })
        super(CartpoleDqnActorAgent, self).__init__(model, plugin_cfg)


class CartpoleDqnEvaluateAgent(BaseAgent):
    def __init__(self, model):
        plugin_cfg = OrderedDict({
            'argmax_sample': {},
            'grad': {
                'enable_grad': False
            },
        })
        super(CartpoleDqnEvaluateAgent, self).__init__(model, plugin_cfg)


class CartpoleDqnLearner(BaseLearner):
    _name = "CartpoleDqnLearner"

    def _setup_agent(self):
        env_info = CartpoleEnv(self._cfg.env).info()
        model = FCDQN(env_info.obs_space.shape, env_info.act_space.shape, dueling=self._cfg.learner.dqn.dueling)
        if self._cfg.learner.use_cuda:
            model.cuda()
        self._agent = CartpoleDqnLearnerAgent(model, is_double=self._cfg.learner.dqn.is_double)
        self._agent.mode(train=True)
        if self._agent.is_double:
            self._agent.target_mode(train=True)

    def _setup_computation_graph(self):
        self._computation_graph = CartpoleDqnGraph(self._cfg.learner)

    def _setup_data_source(self):
        # set in SingleMachineRunner
        pass


class CartpoleRunner(SingleMachineRunner):
    def _setup_env(self):
        env_num = self.cfg.env.env_num
        self.env = SubprocessEnvManager(CartpoleEnv, env_cfg=[self.cfg.env for _ in range(env_num)], env_num=env_num)

    def _setup_learner(self):
        self.learner = CartpoleDqnLearner(self.cfg)

    def _setup_actor_agent(self):
        self.actor_agent = CartpoleDqnActorAgent(copy.deepcopy(self.learner.agent.model))
        self.actor_agent.mode(train=False)

    def _setup_evaluate_agent(self):
        self.evaluate_agent = CartpoleDqnEvaluateAgent(copy.deepcopy(self.learner.agent.model))
        self.evaluate_agent.mode(train=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default="./cartpole_dqn_default_config.yaml")
    args = parser.parse_known_args()[0]
    runner = CartpoleRunner(read_config(args.config_path))
    runner.run()
