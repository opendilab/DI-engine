from nervex.model import ConvDQN, FCDQN
from nervex.worker.learner import BaseLearner
from nervex.torch_utils import CudaFetcher
from nervex.data import default_collate
from app_zoo.mujoco.envs import MujocoEnv
from app_zoo.mujoco.computation_graph.mujoco_dqn_computation_graph import MujocoDqnGraph
from .mujoco_agent import MujocoDqnLearnerAgent


class MujocoDqnLearner(BaseLearner):
    _name = "MujocoDqnLearner"

    def _setup_agent(self) -> None:
        env_info = MujocoEnv(self._cfg.env).info()
        print(env_info)
        # model = ConvDQN(env_info.obs_space.shape, env_info.act_space.shape, dueling=self._cfg.learner.dqn.dueling)
        model = FCDQN(env_info.obs_space.shape, env_info.act_space.shape, dueling=self._cfg.learner.dqn.dueling)
        print(model)
        if self._cfg.learner.use_cuda:
            model.cuda()
        self._agent = MujocoDqnLearnerAgent(model, is_double=self._cfg.learner.dqn.is_double)
        self._agent.mode(train=True)
        if self._agent.is_double:
            self._agent.target_mode(train=True)

    def _setup_computation_graph(self) -> None:
        self._computation_graph = MujocoDqnGraph(self._cfg.learner)
