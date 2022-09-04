from dizoo.classic_control.mountain_car.config.mtcar_rainbow_config import mtcar_rainbow_config, mtcar_rainbow_create_config
from ding.config import compile_config
from dizoo.classic_control.mountain_car.envs.mtcar_env import MountainCarEnv
from ding.envs import BaseEnvManagerV2
from ding.model import RainbowDQN
from ding.policy import RainbowDQNPolicy
from ding.data import DequeBuffer
from ding.data.buffer.middleware import PriorityExperienceReplay
from ding.framework import task
from ding.framework.context import OnlineRLContext
from ding.framework.middleware import OffPolicyLearner, StepCollector, interaction_evaluator, data_pusher, eps_greedy_handler, CkptSaver, nstep_reward_enhancer


def main():
    """
    This is an example of the Rainbow algorithm implemented on the Mountain Car (discrete action space) environment.
    """

    # Config
    cfg = compile_config(mtcar_rainbow_config, create_cfg=mtcar_rainbow_create_config, auto=True)

    # Environment
    # Get number of collector & evaluator env
    coll_env_num, eval_env_num = cfg.env.collector_env_num, cfg.env.evaluator_env_num
    # Instantiate collector and evaluator environment
    coll_env = BaseEnvManagerV2(env_fn=[lambda: MountainCarEnv() for _ in range(coll_env_num)], cfg=cfg.env.manager)
    eval_env = BaseEnvManagerV2(env_fn=[lambda: MountainCarEnv() for _ in range(eval_env_num)], cfg=cfg.env.manager)

    # Policy
    model = RainbowDQN(**cfg.policy.model)
    buffer_ = DequeBuffer(size=cfg.policy.other.replay_buffer.replay_buffer_size)
    buffer_.use(PriorityExperienceReplay(buffer_, IS_weight=True))
    policy = RainbowDQNPolicy(cfg.policy, model=model)

    # Pipeline
    with task.start(async_mode=False, ctx=OnlineRLContext()):
        # Evaluating, we place it on the first place to get the score of the random model as a benchmark value
        task.use(interaction_evaluator(cfg, policy.eval_mode, eval_env))
        # Decay probability of explore-exploit
        task.use(eps_greedy_handler(cfg))
        # Collect environmental data
        task.use(StepCollector(cfg, policy.collect_mode, coll_env))
        # Enable nstep
        task.use(nstep_reward_enhancer(cfg))
        # Push data to buffer
        task.use(data_pusher(cfg, buffer_))
        # Train the model
        task.use(OffPolicyLearner(cfg, policy.learn_mode, buffer_))
        # Save the model
        task.use(CkptSaver(cfg, policy, train_freq=100))
        # In the evaluation process, if the model is found to have exceeded the convergence value, it will end early here
        task.run()

    return


if __name__ == '__main__':
    main()
