from nervex.model import ConvValueAC
from nervex.worker.learner import BaseLearner, register_learner
from nervex.worker.agent import create_ac_learner_agent
from nervex.utils import DistModule
from app_zoo.atari.envs import AtariEnv
from app_zoo.atari.computation_graph.atari_ppo_computation_graph import AtariPpoGraph
from app_zoo.atari.computation_graph.atari_a2c_computation_graph import AtariA2CGraph


class AtariPpoLearner(BaseLearner):
    _name = "AtariPpoLearner"

    def _setup_agent(self) -> None:
        env_info = AtariEnv(self._cfg.env).info()
        model = ConvValueAC(env_info.obs_space.shape, env_info.act_space.shape, self._cfg.model.embedding_dim)
        if self._cfg.learner.use_cuda:
            model.cuda()
        if self._cfg.learner.use_distributed:
            model = DistModule(model)
        self._agent = create_ac_learner_agent(model)
        self._agent.mode(train=True)

    def _setup_computation_graph(self) -> None:
        self._computation_graph = AtariPpoGraph(self._cfg.learner)


class AtariA2CLearner(AtariPpoLearner):
    _name = "AtariA2CLearner"

    def _setup_computation_graph(self) -> None:
        self._computation_graph = AtariA2CGraph(self._cfg.learner)


register_learner('atari_ppo', AtariPpoLearner)
register_learner('atari_a2c', AtariA2CLearner)
