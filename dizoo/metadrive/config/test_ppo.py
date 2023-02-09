from easydict import EasyDict
from functools import partial
from tensorboardX import SummaryWriter
import torch 
from ding.envs import BaseEnvManager, SyncSubprocessEnvManager
from ding.config import compile_config
from ding.model.template import  VAC
from ding.policy import PPOPolicy
from ding.worker import SampleSerialCollector, InteractionSerialEvaluator, BaseLearner
from dizoo.metadrive.env.drive_env import MetaDrivePPOOriginEnv
from dizoo.metadrive.env.drive_wrapper import DriveEnvWrapper


# ckpt dir: 
model_dir = None
metadrive_basic_config = dict(
    exp_name='test_ppo_metadrive',
    env=dict(
        metadrive=dict(
            use_render = True,
            traffic_density=0.10,
            map = 'XSOS',
            horizon = 4000,
            driving_reward = 1.0,
            speed_reward = 0.10,
            out_of_road_penalty = 40.0,
            crash_vehicle_penalty = 40.0,
            decision_repeat=20,
            use_lateral_reward = False,
            out_of_route_done = True,
            ),
        manager=dict(
            shared_memory=False,
            max_retry=2,
            context='spawn',
        ),
        n_evaluator_episode=16,
        stop_value=99999,
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
            entropy_weight = 0.001,
            value_weight=0.5,
            clip_ratio = 0.02,
            adv_norm=False,
            value_norm=True,
            grad_clip_value=10,
        ),
        collect=dict(
            n_sample=1000,
        ),
        eval=dict(
            evaluator=dict(
                eval_freq=1000,
            ),
        ),
    ),
)
main_config = EasyDict(metadrive_basic_config)

def wrapped_env(env_cfg, wrapper_cfg=None):
    return DriveEnvWrapper(MetaDrivePPOOriginEnv(env_cfg), wrapper_cfg)

def main(cfg):
    cfg = compile_config(
        cfg, BaseEnvManager, PPOPolicy, BaseLearner, SampleSerialCollector, InteractionSerialEvaluator
    )
    collector_env_num, evaluator_env_num = cfg.env.collector_env_num, cfg.env.evaluator_env_num
    evaluator_env = BaseEnvManager(
        env_fn=[partial(wrapped_env, cfg.env.metadrive) for _ in range(evaluator_env_num)],
        cfg=cfg.env.manager,
    )
    model = VAC(**cfg.policy.model)
    policy = PPOPolicy(cfg.policy, model=model)
    if model_dir is not None:
        policy._load_state_dict_collect(torch.load(model_dir, map_location='cpu'))
    tb_logger = SummaryWriter('./log/{}/'.format(cfg.exp_name))
    learner = BaseLearner(cfg.policy.learn.learner, policy.learn_mode, tb_logger, exp_name=cfg.exp_name)
    evaluator = InteractionSerialEvaluator(
        cfg.policy.eval.evaluator, evaluator_env, policy.eval_mode, tb_logger, exp_name=cfg.exp_name
    )
    learner.call_hook('before_run')
    stop, rate = evaluator.eval()
    evaluator.close()
    learner.close()


if __name__ == '__main__':
    main(main_config)