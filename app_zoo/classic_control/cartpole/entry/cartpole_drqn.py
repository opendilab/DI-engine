import argparse
import torch
import copy
from collections import OrderedDict

from nervex.computation_graph import BaseCompGraph
from nervex.entry.base_single_machine import SingleMachineRunner
from nervex.model import FCDRQN
from nervex.data import timestep_collate, AsyncDataLoader
from nervex.rl_utils import q_nstep_td_data, q_nstep_td_error, q_nstep_td_error_with_rescale
from nervex.utils import read_config
from nervex.worker import BaseLearner, SubprocessEnvManager
from nervex.worker.agent import create_drqn_learner_agent, create_drqn_actor_agent, create_drqn_evaluator_agent
from app_zoo.classic_control.cartpole.envs import CartPoleEnv


class CartPoleDrqnGraph(BaseCompGraph):

    def __init__(self, cfg):
        self._gamma = cfg.dqn.discount_factor
        self._nstep = cfg.dqn.nstep
        self._burnin_step = cfg.dqn.burnin_step
        self._use_value_rescale = cfg.dqn.get("value_rescale", False)

    def forward(self, data, agent):
        obs, reward = data['obs'], data['reward']
        assert len(obs) == 2 * self._nstep + self._burnin_step, obs.shape
        action = data.get('action')[:self._nstep]
        done = data.get('done')[:self._nstep].float()
        weights = data.get('weights', None)

        agent.reset(state=data['prev_state'][0])

        bs = self._burnin_step
        if agent.is_double:
            agent.target_reset(state=data['prev_state'][0])
            if bs != 0:
                with torch.no_grad():
                    inputs = {'obs': obs[:bs], 'enable_fast_timestep': True}
                    _ = agent.forward(inputs)
                    _ = agent.target_forward(inputs)
            estimate_obs = obs[bs:bs + self._nstep]
            target_obs = obs[bs:]
            inputs = {'obs': estimate_obs, 'enable_fast_timestep': True}
            q_value = agent.forward(inputs)['logit']
            next_inputs = {'obs': target_obs, 'enable_fast_timestep': True}
            target_q_value = agent.target_forward(next_inputs)['logit'][self._nstep:]
        else:
            if bs != 0:
                with torch.no_grad():
                    inputs = {'obs': obs[:bs], 'enable_fast_timestep': True}
                    _ = agent.forward(inputs)
            inputs = {'obs': obs[bs:], 'enable_fast_timestep': True}
            logit = agent.forward(inputs)['logit']
            q_value = logit[:self._nstep]
            target_q_value = logit[self._nstep:]

        loss = []
        for t in range(self._nstep):
            data = q_nstep_td_data(q_value[t], target_q_value[t], action[t], reward[t:t + self._nstep], done[t])
            if self._use_value_rescale:
                loss.append(q_nstep_td_error_with_rescale(data, self._gamma, self._nstep, weights))
            else:
                loss.append(q_nstep_td_error(data, self._gamma, self._nstep, weights))
        loss = sum(loss) / (len(loss) + 1e-8)
        if agent.is_double:
            agent.target_update(agent.state_dict()['model'])
        return {'total_loss': loss}

    def __repr__(self):
        return "CartPoleDrqnGraph"


class CartPoleDrqnLearner(BaseLearner):
    _name = "CartPoleDrqnLearner"

    def _setup_agent(self):
        env_info = CartPoleEnv(self._cfg.env).info()
        model = FCDRQN(env_info.obs_space.shape, env_info.act_space.shape, dueling=self._cfg.learner.dqn.dueling)
        if self._cfg.learner.use_cuda:
            model.cuda()
        self._agent = create_drqn_learner_agent(
            model, state_num=self._cfg.learner.data.batch_size, is_double=self._cfg.learner.dqn.is_double
        )
        self._agent.mode(train=True)
        if self._agent.is_double:
            self._agent.target_mode(train=True)

    def _setup_computation_graph(self):
        self._computation_graph = CartPoleDrqnGraph(self._cfg.learner)

    def _setup_dataloader(self):
        cfg = self._cfg.learner.data
        self._dataloader = AsyncDataLoader(
            self.get_data, cfg.batch_size, self._device, cfg.chunk_size, timestep_collate, cfg.num_workers
        )


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

        eval_env_num = self.cfg.evaluator.env_num
        evaluate_env_cfg = copy.deepcopy(self.cfg.env)
        self.evaluate_env = SubprocessEnvManager(
            CartPoleEnv,
            env_cfg=[evaluate_env_cfg for _ in range(eval_env_num)],
            env_num=eval_env_num,
            episode_num=self.cfg.evaluator.episode_num
        )

    def _setup_learner(self):
        self.learner = CartPoleDrqnLearner(self.cfg)

    def _setup_agent(self):
        self.actor_agent = create_drqn_actor_agent(
            copy.deepcopy(self.learner.agent.model), state_num=self.cfg.actor.env_num
        )
        self.actor_agent.mode(train=False)
        self.evaluator_agent = create_drqn_evaluator_agent(
            copy.deepcopy(self.learner.agent.model), state_num=self.cfg.evaluator.env_num
        )
        self.evaluator_agent.mode(train=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default="./cartpole_drqn_default_config.yaml")
    args = parser.parse_known_args()[0]
    runner = CartPoleRunner(read_config(args.config_path))
    runner.run()
