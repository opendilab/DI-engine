import argparse
import copy
from collections import OrderedDict

from nervex.computation_graph import BaseCompGraph
from nervex.entry.base_single_machine import SingleMachineRunner
from nervex.model import FCDQN
from nervex.rl_utils import q_1step_td_data, q_1step_td_error
from nervex.utils import read_config
from nervex.worker import BaseLearner, SubprocessEnvManager
from nervex.worker.agent import create_dqn_learner_agent, create_dqn_actor_agent, create_dqn_evaluator_agent
from app_zoo.classic_control.cartpole.envs import CartPoleEnv


class CartPoleDqnGraph(BaseCompGraph):

    def __init__(self, cfg):
        self._gamma = cfg.dqn.discount_factor

    def forward(self, data, agent):
        obs = data.get('obs')
        nextobs = data.get('next_obs')
        reward = data.get('reward')
        action = data.get('action')
        terminate = data.get('done').float()
        weights = data.get('weights', None)

        q_value = agent.forward(obs)['logit']
        if agent.is_double:
            target_q_value = agent.target_forward(nextobs)['logit']
        else:
            target_q_value = agent.forward(nextobs)['logit']

        data = q_1step_td_data(q_value, target_q_value, action, reward, terminate)
        loss = q_1step_td_error(data, self._gamma, weights)
        if agent.is_double:
            agent.target_update(agent.state_dict()['model'])
        return {'total_loss': loss}

    def __repr__(self):
        return "CartPoleDqnGraph"


class CartPoleDqnLearner(BaseLearner):
    _name = "CartPoleDqnLearner"

    def _setup_agent(self):
        env_info = CartPoleEnv(self._cfg.env).info()
        model = FCDQN(env_info.obs_space.shape, env_info.act_space.shape, dueling=self._cfg.learner.dqn.dueling)
        if self._cfg.learner.use_cuda:
            model.cuda()
        self._agent = create_dqn_learner_agent(model, is_double=self._cfg.learner.dqn.is_double)
        self._agent.mode(train=True)
        if self._agent.is_double:
            self._agent.target_mode(train=True)

    def _setup_computation_graph(self):
        self._computation_graph = CartPoleDqnGraph(self._cfg.learner)


class CartPoleRunner(SingleMachineRunner):

    def _setup_env(self):
        actor_env_num = self.cfg.actor.env_num
        actor_env_cfg = copy.deepcopy(self.cfg.env)
        self.actor_env = SubprocessEnvManager(
            CartPoleEnv,
            env_cfg=[actor_env_cfg for _ in range(actor_env_num)],
            env_num=actor_env_num,
            episode_num=self.cfg.actor.episode_num
        )
        self.actor_env.launch()

        eval_env_num = self.cfg.evaluator.env_num
        evaluate_env_cfg = copy.deepcopy(self.cfg.env)
        self.evaluate_env = SubprocessEnvManager(
            CartPoleEnv,
            env_cfg=[evaluate_env_cfg for _ in range(eval_env_num)],
            env_num=eval_env_num,
            episode_num=self.cfg.evaluator.episode_num
        )
        self.evaluate_env.launch()

    def _setup_learner(self):
        self.learner = CartPoleDqnLearner(self.cfg)

    def _setup_agent(self):
        self.actor_agent = create_dqn_actor_agent(copy.deepcopy(self.learner.agent.model))
        self.actor_agent.mode(train=False)
        self.evaluator_agent = create_dqn_evaluator_agent(copy.deepcopy(self.learner.agent.model))
        self.evaluator_agent.mode(train=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default="./cartpole_dqn_default_config.yaml")
    args = parser.parse_known_args()[0]
    runner = CartPoleRunner(read_config(args.config_path))
    runner.run()
