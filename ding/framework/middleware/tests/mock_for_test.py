from typing import TYPE_CHECKING, Union, Any, List, Callable, Dict, Optional
from collections import namedtuple
import random
import torch
import treetensor.numpy as tnp
from easydict import EasyDict
from unittest.mock import Mock

from ding.torch_utils import to_device
from ding.league.player import PlayerMeta
from ding.league.v2 import BaseLeague, Job
from ding.framework.storage import FileStorage

obs_dim = [2, 2]
action_space = 1
env_num = 2

CONFIG = dict(
    seed=0,
    policy=dict(
        learn=dict(
            update_per_collect=4,
            batch_size=8,
            learner=dict(hook=dict(log_show_after_iter=10), ),
        ),
        collect=dict(
            n_sample=16,
            unroll_len=1,
            n_episode=16,
        ),
        eval=dict(evaluator=dict(eval_freq=10), ),
        other=dict(eps=dict(
            type='exp',
            start=0.95,
            end=0.1,
            decay=10000,
        ), ),
    ),
    env=dict(
        n_evaluator_episode=5,
        stop_value=2.0,
    ),
)
CONFIG = EasyDict(CONFIG)


class MockPolicy(Mock):

    def __init__(self) -> None:
        super(MockPolicy, self).__init__()
        self.action_space = action_space
        self.obs_dim = obs_dim

    def reset(self, data_id: Optional[List[int]] = None) -> None:
        return

    def forward(self, data: dict, **kwargs) -> dict:
        res = {}
        for i, v in data.items():
            res[i] = {'action': torch.sum(v)}
        return res

    def process_transition(self, obs: Any, model_output: dict, timestep: namedtuple) -> dict:
        transition = {
            'obs': torch.rand(self.obs_dim),
            'next_obs': torch.rand(self.obs_dim),
            'action': torch.zeros(self.action_space),
            'logit': 1.0,
            'value': 2.0,
            'reward': 0.1,
            'done': True,
        }
        return transition


class MockEnv(Mock):

    def __init__(self) -> None:
        super(MockEnv, self).__init__()
        self.env_num = env_num
        self.obs_dim = obs_dim
        self.closed = False
        self._reward_grow_indicator = 1

    @property
    def ready_obs(self) -> tnp.array:
        return tnp.stack([
            torch.zeros(self.obs_dim),
            torch.ones(self.obs_dim),
        ])

    def seed(self, seed: Union[Dict[int, int], List[int], int], dynamic_seed: bool = None) -> None:
        return

    def launch(self, reset_param: Optional[Dict] = None) -> None:
        return

    def reset(self, reset_param: Optional[Dict] = None) -> None:
        return

    def step(self, actions: tnp.ndarray) -> List[tnp.ndarray]:
        timesteps = []
        for i in range(self.env_num):
            timestep = dict(
                obs=torch.rand(self.obs_dim),
                reward=1.0,
                done=True,
                info={'final_eval_reward': self._reward_grow_indicator * 1.0},
                env_id=i,
            )
            timesteps.append(tnp.array(timestep))
        self._reward_grow_indicator += 1  # final_eval_reward will increase as step method is called
        return timesteps


class MockHerRewardModel(Mock):

    def __init__(self) -> None:
        super(MockHerRewardModel, self).__init__()
        self.episode_size = 8
        self.episode_element_size = 4

    def estimate(self, episode: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [[episode[0] for _ in range(self.episode_element_size)]]


class MockLeague(BaseLeague):

    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self.update_payoff_cnt = 0
        self.update_active_player_cnt = 0
        self.create_historical_player_cnt = 0
        self.get_job_info_cnt = 0

    def update_payoff(self, job):
        self.update_payoff_cnt += 1

    def update_active_player(self, meta):
        self.update_active_player_cnt += 1

    def create_historical_player(self, meta):
        self.create_historical_player_cnt += 1

    def get_job_info(self, player_id):
        self.get_job_info_cnt += 1
        other_players = [i for i in self.active_players_ids if i != player_id]
        another_palyer = random.choice(other_players)
        return Job(
            launch_player=player_id,
            players=[
                PlayerMeta(player_id=player_id, checkpoint=FileStorage(path=None), total_agent_step=0),
                PlayerMeta(player_id=another_palyer, checkpoint=FileStorage(path=None), total_agent_step=0)
            ]
        )


class MockLogger():

    def add_scalar(*args):
        pass

    def close(*args):
        pass

    def flush(*args):
        pass


league_cfg = EasyDict(
    {
        'env': {
            'manager': {
                'episode_num': 100000,
                'max_retry': 1000,
                'retry_type': 'renew',
                'auto_reset': True,
                'step_timeout': None,
                'reset_timeout': None,
                'retry_waiting_time': 0.1,
                'cfg_type': 'BaseEnvManagerDict',
                'shared_memory': False,
                'return_original_data': True
            },
            'collector_env_num': 1,
            'evaluator_env_num': 1,
            'n_evaluator_episode': 100,
            'env_type': 'prisoner_dilemma',
            'stop_value': [-10.1, -5.05]
        },
        'policy': {
            'model': {
                'obs_shape': 2,
                'action_shape': 2,
                'action_space': 'discrete',
                'encoder_hidden_size_list': [32, 32],
                'critic_head_hidden_size': 32,
                'actor_head_hidden_size': 32,
                'share_encoder': False
            },
            'learn': {
                'learner': {
                    'train_iterations': 1000000000,
                    'dataloader': {
                        'num_workers': 0
                    },
                    'log_policy': False,
                    'hook': {
                        'load_ckpt_before_run': '',
                        'log_show_after_iter': 100,
                        'save_ckpt_after_iter': 10000,
                        'save_ckpt_after_run': True
                    },
                    'cfg_type': 'BaseLearnerDict'
                },
                'multi_gpu': False,
                'epoch_per_collect': 10,
                'batch_size': 16,
                'learning_rate': 1e-05,
                'value_weight': 0.5,
                'entropy_weight': 0.0,
                'clip_ratio': 0.2,
                'adv_norm': True,
                'value_norm': True,
                'ppo_param_init': True,
                'grad_clip_type': 'clip_norm',
                'grad_clip_value': 0.5,
                'ignore_done': False,
                'update_per_collect': 3,
                'scheduler': {
                    'schedule_flag': False,
                    'schedule_mode': 'reduce',
                    'factor': 0.005,
                    'change_range': [0, 1],
                    'threshold': 0.5,
                    'patience': 50
                }
            },
            'collect': {
                'collector': {
                    'deepcopy_obs': False,
                    'transform_obs': False,
                    'collect_print_freq': 100,
                    'get_train_sample': True,
                    'cfg_type': 'BattleEpisodeSerialCollectorDict'
                },
                'discount_factor': 1.0,
                'gae_lambda': 1.0,
                'n_episode': 1,
                'n_rollout_samples': 64,
                'n_sample': 64,
                'unroll_len': 1
            },
            'eval': {
                'evaluator': {
                    'eval_freq': 50,
                    'cfg_type': 'BattleInteractionSerialEvaluatorDict',
                    'stop_value': [-10.1, -5.05],
                    'n_episode': 100
                }
            },
            'other': {
                'replay_buffer': {
                    'type': 'naive',
                    'replay_buffer_size': 10000,
                    'deepcopy': False,
                    'enable_track_used_data': False,
                    'periodic_thruput_seconds': 60,
                    'cfg_type': 'NaiveReplayBufferDict'
                },
                'league': {
                    'player_category': ['default'],
                    'path_policy': 'league_demo/ckpt',
                    'active_players': {
                        'main_player': 2
                    },
                    'main_player': {
                        'one_phase_step': 10,  # 20
                        'branch_probs': {
                            'pfsp': 0.0,
                            'sp': 1.0
                        },
                        'strong_win_rate': 0.7
                    },
                    'main_exploiter': {
                        'one_phase_step': 200,
                        'branch_probs': {
                            'main_players': 1.0
                        },
                        'strong_win_rate': 0.7,
                        'min_valid_win_rate': 0.3
                    },
                    'league_exploiter': {
                        'one_phase_step': 200,
                        'branch_probs': {
                            'pfsp': 1.0
                        },
                        'strong_win_rate': 0.7,
                        'mutate_prob': 0.5
                    },
                    'use_pretrain': False,
                    'use_pretrain_init_historical': False,
                    'payoff': {
                        'type': 'battle',
                        'decay': 0.99,
                        'min_win_rate_games': 8
                    },
                    'metric': {
                        'mu': 0,
                        'sigma': 8.333333333333334,
                        'beta': 4.166666666666667,
                        'tau': 0.0,
                        'draw_probability': 0.02
                    }
                }
            },
            'type': 'ppo',
            'cuda': False,
            'on_policy': True,
            'priority': False,
            'priority_IS_weight': False,
            'recompute_adv': True,
            'action_space': 'discrete',
            'nstep_return': False,
            'multi_agent': False,
            'transition_with_policy_data': True,
            'cfg_type': 'PPOPolicyDict'
        },
        'exp_name': 'league_demo',
        'seed': 0
    }
)
