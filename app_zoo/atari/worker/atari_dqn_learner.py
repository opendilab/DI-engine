from nervex.model import ConvDQN
from nervex.worker.learner import BaseLearner
from nervex.torch_utils import CudaFetcher
from nervex.data import default_collate
from app_zoo.atari.envs import AtariEnv
from app_zoo.atari.computation_graph.atari_dqn_computation_graph import AtariDqnGraph
from .atari_agent import AtariDqnLearnerAgent


class AtariDqnLearner(BaseLearner):
    _name = "AtariDqnLearner"

    def _setup_agent(self) -> None:
        env_info = AtariEnv(self._cfg.env).info()
        model = ConvDQN(env_info.obs_space.shape, env_info.act_space.shape, dueling=self._cfg.learner.dqn.dueling)
        if self._cfg.learner.use_cuda:
            model.cuda()
        self._agent = AtariDqnLearnerAgent(model, is_double=self._cfg.learner.dqn.is_double)
        self._agent.mode(train=True)
        if self._agent.is_double:
            self._agent.target_mode(train=True)

    def _setup_computation_graph(self) -> None:
        self._computation_graph = AtariDqnGraph(self._cfg.learner)

    def _setup_data_source(self):
        self._collate_fn = default_collate
        batch_size = self._cfg.learner.batch_size

        def iterator():
            while True:
                data = self.get_data(batch_size)
                yield self._collate_fn(data)

        self._data_source = iterator()
        if self._use_cuda:
            self._data_source = CudaFetcher(self._data_source, device=self._device, sleep=0.01)
