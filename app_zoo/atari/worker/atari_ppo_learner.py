from nervex.model import ConvValueAC
from nervex.worker.learner import BaseLearner
from app_zoo.atari.envs import AtariEnv
from app_zoo.atari.computation_graph.atari_ppo_computation_graph import AtariPpoGraph
from .atari_agent import AtariPpoLearnerAgent


class AtariPpoLearner(BaseLearner):
    _name = "AtariPpoLearner"

    def _setup_agent(self) -> None:
        env_info = AtariEnv(self._cfg.env).info()
        model = ConvValueAC(env_info.obs_space.shape, env_info.act_space.shape, self._cfg.model.embedding_dim)
        if self._cfg.learner.use_cuda:
            model.cuda()
        self._agent = AtariPpoLearnerAgent(model)
        self._agent.mode(train=True)

    def _setup_computation_graph(self) -> None:
        self._computation_graph = AtariPpoGraph(self._cfg.learner)
