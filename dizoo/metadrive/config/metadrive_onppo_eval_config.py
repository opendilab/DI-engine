from easydict import EasyDict
from functools import partial
from tensorboardX import SummaryWriter
import torch
from ding.envs import BaseEnvManager, SyncSubprocessEnvManager
from ding.config import compile_config
from ding.model.template import VAC
from ding.policy import PPOPolicy
from ding.worker import SampleSerialCollector, InteractionSerialEvaluator, BaseLearner
from dizoo.metadrive.env.drive_env import MetaDrivePPOOriginEnv
from dizoo.metadrive.env.drive_wrapper import DriveEnvWrapper

# Load the trained model from this direction, if None, it will initialize from scratch
model_dir = None
metadrive_basic_config = dict(
    exp_name='metadrive_onppo_eval_seed0',
    env=dict(
        metadrive=dict(
            use_render=True,
            traffic_density=0.10,  # Density of vehicles occupying the roads, range in [0,1]
            map='XSOS',  # Int or string: an easy way to fill map_config
            horizon=4000,  # Max step number
            driving_reward=1.0,  # Reward to encourage agent to move forward.
            speed_reward=0.10,  # Reward to encourage agent to drive at a high speed
            use_lateral_reward=False,  # reward for lane keeping
            out_of_road_penalty=40.0,  # Penalty to discourage driving out of road
            crash_vehicle_penalty=40.0,  # Penalty to discourage collision
            decision_repeat=20,  # Reciprocal of decision frequency
            out_of_route_done=True,  # Game over if driving out of road
            show_bird_view=False,  # Only used to evaluate, whether to draw five channels of bird-view image
        ),
        manager=dict(
            shared_memory=False,
            max_retry=2,
            context='spawn',
        ),
        n_evaluator_episode=16,
        stop_value=255,
        collector_env_num=1,
        evaluator_env_num=1,
    ),
    policy=dict(
        cuda=True,
        action_space='continuous',
        model=dict(
            obs_shape=[5, 84, 84],
            action_shape=2,
            action_space='continuous',
            bound_type='tanh',
            encoder_hidden_size_list=[128, 128, 64],
        ),
        learn=dict(
            epoch_per_collect=10,
            batch_size=64,
            learning_rate=3e-4,
            entropy_weight=0.001,
            value_weight=0.5,
            clip_ratio=0.02,
            adv_norm=False,
            value_norm=True,
            grad_clip_value=10,
        ),
        collect=dict(n_sample=1000, ),
        eval=dict(evaluator=dict(eval_freq=1000, ), ),
    ),
)
main_config = EasyDict(metadrive_basic_config)


def wrapped_env(env_cfg, wrapper_cfg=None):
    return DriveEnvWrapper(MetaDrivePPOOriginEnv(env_cfg), wrapper_cfg)


def main(cfg):
    cfg = compile_config(cfg, BaseEnvManager, PPOPolicy, BaseLearner, SampleSerialCollector, InteractionSerialEvaluator)
    evaluator_env_num = cfg.env.evaluator_env_num
    show_bird_view = cfg.env.metadrive.show_bird_view
    wrapper_cfg = {'show_bird_view': show_bird_view}
    evaluator_env = BaseEnvManager(
        env_fn=[partial(wrapped_env, cfg.env.metadrive, wrapper_cfg) for _ in range(evaluator_env_num)],
        cfg=cfg.env.manager,
    )
    model = VAC(**cfg.policy.model)
    policy = PPOPolicy(cfg.policy, model=model)
    if model_dir is not None:
        policy._load_state_dict_collect(torch.load(model_dir, map_location='cpu'))
    tb_logger = SummaryWriter('./log/{}/'.format(cfg.exp_name))
    evaluator = InteractionSerialEvaluator(
        cfg.policy.eval.evaluator, evaluator_env, policy.eval_mode, tb_logger, exp_name=cfg.exp_name
    )
    stop, rate = evaluator.eval()
    evaluator.close()


if __name__ == '__main__':
    main(main_config)
