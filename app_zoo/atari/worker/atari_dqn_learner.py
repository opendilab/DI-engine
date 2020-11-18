from nervex.model import ConvDQN
from nervex.worker.learner import BaseLearner
from nervex.worker.agent import create_dqn_learner_agent
from app_zoo.atari.envs import AtariEnv
from app_zoo.atari.computation_graph.atari_dqn_computation_graph import AtariDqnGraph


class AtariDqnLearner(BaseLearner):
    _name = "AtariDqnLearner"

    def _setup_agent(self) -> None:
        env_info = AtariEnv(self._cfg.env).info()
        model = ConvDQN(env_info.obs_space.shape, env_info.act_space.shape, dueling=self._cfg.learner.dqn.dueling)
        if self._cfg.learner.use_cuda:
            model.cuda()
        self._agent = create_dqn_learner_agent(model, is_double=self._cfg.learner.dqn.is_double)
        self._agent.mode(train=True)
        if self._agent.is_double:
            self._agent.target_mode(train=True)

    def _setup_computation_graph(self) -> None:
        self._computation_graph = AtariDqnGraph(self._cfg.learner)
