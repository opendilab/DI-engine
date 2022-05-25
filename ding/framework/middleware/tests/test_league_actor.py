from ding.model import DQN
from ding.policy import DQNPolicy
from ding.envs import DingEnvWrapper, BaseEnvManagerV2
from ding.data import DequeBuffer
from ding.config import compile_config
from ding.framework import task
from ding.framework.context import OnlineRLContext
from ding.framework.middleware import LeagueActor, Job
from ding.utils import set_pkg_seed
from dizoo.classic_control.cartpole.config.cartpole_dqn_config import main_config, create_config
from dizoo.distar.envs.distar_env import DIStarEnv
from distar.ctools.utils import read_config


def main():
    distar_cfg = read_config('C:/Users/hjs/DI-engine/dizoo/distar/envs/test_distar_config.yaml')

    cfg = compile_config(main_config, create_cfg=create_config, auto=True)
    with task.start(async_mode=False, ctx=OnlineRLContext()):
        collector_env = BaseEnvManagerV2(
            env_fn=[lambda: DingEnvWrapper(DIStarEnv(distar_cfg)) for _ in range(cfg.env.collector_env_num)],
            cfg=cfg.env.manager
        )

        set_pkg_seed(cfg.seed, use_cuda=cfg.policy.cuda)

        model = DQN(**cfg.policy.model)
        policy = DQNPolicy(cfg.policy, model=model)

        task.use(LeagueActor(cfg, policy.collect_mode, collector_env))
        task.run()


if __name__ == "__main__":
    main()