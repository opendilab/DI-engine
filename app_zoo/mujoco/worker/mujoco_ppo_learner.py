from nervex.model import ConvValueAC
from nervex.worker.learner import BaseLearner
from nervex.torch_utils import CudaFetcher
from nervex.data import default_collate
from app_zoo.mujoco.envs import MujocoEnv
from app_zoo.mujoco.computation_graph.mujoco_ppo_computation_graph import MujocoPpoGraph
from .mujoco_agent import MujocoPpoLearnerAgent


class MujocoPpoLearner(BaseLearner):
    _name = "MujocoPpoLearner"

    def _setup_agent(self) -> None:
        env_info = MujocoEnv(self._cfg.env).info()
        model = ConvValueAC(env_info.obs_space.shape, env_info.act_space.shape, self._cfg.model.embedding_dim)
        if self._cfg.learner.use_cuda:
            model.cuda()
        self._agent = MujocoPpoLearnerAgent(model)
        self._agent.mode(train=True)

    def _setup_computation_graph(self) -> None:
        self._computation_graph = MujocoPpoGraph(self._cfg.learner)

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
