from typing import Optional, Any, List
from collections import namedtuple
from easydict import EasyDict
import numpy as np
import torch

from ding.envs import BaseEnvManager
from ding.utils import build_logger, EasyTimer, SERIAL_COLLECTOR_REGISTRY
from ding.torch_utils import to_tensor, to_ndarray
from .base_serial_collector import ISerialCollector, CachePool, TrajBuffer, INF, to_tensor_transitions
from ding.rl_utils.mcts.game import GameHistory
from ding.rl_utils.mcts.utils import select_action, prepare_observation_lst
from torch.nn import L1Loss
from ding.envs import BaseEnvTimestep


@SERIAL_COLLECTOR_REGISTRY.register('episode_muzero')
class EpisodeSerialCollectorMuZero(ISerialCollector):
    """
    Overview:
        Episode collector(n_episode)
    Interfaces:
        __init__, reset, reset_env, reset_policy, collect, close
    Property:
        envstep
    """

    config = dict(deepcopy_obs=False, transform_obs=False, collect_print_freq=100, get_train_sample=False)

    def __init__(
            self,
            cfg: EasyDict,
            env: BaseEnvManager = None,
            policy: namedtuple = None,
            tb_logger: 'SummaryWriter' = None,  # noqa
            exp_name: Optional[str] = 'default_experiment',
            instance_name: Optional[str] = 'collector',
            replay_buffer: 'replay_buffer' = None,  # noqa
            game_config: 'game_config' = None,  # noqa
    ) -> None:
        """
        Overview:
            Initialization method.
        Arguments:
            - cfg (:obj:`EasyDict`): Config dict
            - env (:obj:`BaseEnvManager`): the subclass of vectorized env_manager(BaseEnvManager)
            - policy (:obj:`namedtuple`): the api namedtuple of collect_mode policy
            - tb_logger (:obj:`SummaryWriter`): tensorboard handle
        """
        self._exp_name = exp_name
        self._instance_name = instance_name
        self._collect_print_freq = cfg.collect_print_freq
        self._deepcopy_obs = cfg.deepcopy_obs
        self._transform_obs = cfg.transform_obs
        self._cfg = cfg
        self._timer = EasyTimer()
        self._end_flag = False

        # MuZero
        self.replay_buffer = replay_buffer
        self.game_config = game_config

        if tb_logger is not None:
            self._logger, _ = build_logger(
                path='./{}/log/{}'.format(self._exp_name, self._instance_name), name=self._instance_name, need_tb=False
            )
            self._tb_logger = tb_logger
        else:
            self._logger, self._tb_logger = build_logger(
                path='./{}/log/{}'.format(self._exp_name, self._instance_name), name=self._instance_name
            )
        self.reset(policy, env)

    def reset_env(self, _env: Optional[BaseEnvManager] = None) -> None:
        """
        Overview:
            Reset the environment.
            If _env is None, reset the old environment.
            If _env is not None, replace the old environment in the collector with the new passed \
                in environment and launch.
        Arguments:
            - env (:obj:`Optional[BaseEnvManager]`): instance of the subclass of vectorized \
                env_manager(BaseEnvManager)
        """
        if _env is not None:
            self._env = _env
            self._env.launch()
            self._env_num = self._env.env_num
        else:
            self._env.reset()

    def reset_policy(self, _policy: Optional[namedtuple] = None) -> None:
        """
        Overview:
            Reset the policy.
            If _policy is None, reset the old policy.
            If _policy is not None, replace the old policy in the collector with the new passed in policy.
        Arguments:
            - policy (:obj:`Optional[namedtuple]`): the api namedtuple of collect_mode policy
        """
        assert hasattr(self, '_env'), "please set env first"
        if _policy is not None:
            self._policy = _policy
            self._default_n_episode = _policy.get_attribute('cfg').collect.get('n_episode', None)
            self._unroll_len = _policy.get_attribute('unroll_len')
            self._on_policy = _policy.get_attribute('cfg').on_policy
            self._traj_len = INF
            self._logger.debug(
                'Set default n_episode mode(n_episode({}), env_num({}), traj_len({}))'.format(
                    self._default_n_episode, self._env_num, self._traj_len
                )
            )
        self._policy.reset()

    def reset(self, _policy: Optional[namedtuple] = None, _env: Optional[BaseEnvManager] = None) -> None:
        """
        Overview:
            Reset the environment and policy.
            If _env is None, reset the old environment.
            If _env is not None, replace the old environment in the collector with the new passed \
                in environment and launch.
            If _policy is None, reset the old policy.
            If _policy is not None, replace the old policy in the collector with the new passed in policy.
        Arguments:
            - policy (:obj:`Optional[namedtuple]`): the api namedtuple of collect_mode policy
            - env (:obj:`Optional[BaseEnvManager]`): instance of the subclass of vectorized \
                env_manager(BaseEnvManager)
        """
        if _env is not None:
            self.reset_env(_env)
        if _policy is not None:
            self.reset_policy(_policy)

        self._obs_pool = CachePool('obs', self._env_num, deepcopy=self._deepcopy_obs)
        self._policy_output_pool = CachePool('policy_output', self._env_num)
        # _traj_buffer is {env_id: TrajBuffer}, is used to store traj_len pieces of transitions
        self._traj_buffer = {env_id: TrajBuffer(maxlen=self._traj_len) for env_id in range(self._env_num)}
        self._env_info = {env_id: {'time': 0., 'step': 0} for env_id in range(self._env_num)}

        self._episode_info = []
        self._total_envstep_count = 0
        self._total_episode_count = 0
        self._total_duration = 0
        self._last_train_iter = 0
        self._end_flag = False
        # double buffering when data is sufficient
        self.trajectory_pool = []
        self.pool_size = 1
        self.gap_step = self.game_config.num_unroll_steps + self.game_config.td_steps
        self.last_model_index = -1

    def _reset_stat(self, env_id: int) -> None:
        """
        Overview:
            Reset the collector's state. Including reset the traj_buffer, obs_pool, policy_output_pool\
                and env_info. Reset these states according to env_id. You can refer to base_serial_collector\
                to get more messages.
        Arguments:
            - env_id (:obj:`int`): the id where we need to reset the collector's state
        """
        self._traj_buffer[env_id].clear()
        self._obs_pool.reset(env_id)
        self._policy_output_pool.reset(env_id)
        self._env_info[env_id] = {'time': 0., 'step': 0}

    @property
    def envstep(self) -> int:
        """
        Overview:
            Print the total envstep count.
        Return:
            - envstep (:obj:`int`): the total envstep count
        """
        return self._total_envstep_count

    def close(self) -> None:
        """
        Overview:
            Close the collector. If end_flag is False, close the environment, flush the tb_logger\
                and close the tb_logger.
        """
        if self._end_flag:
            return
        self._end_flag = True
        self._env.close()
        self._tb_logger.flush()
        self._tb_logger.close()

    def __del__(self) -> None:
        """
        Overview:
            Execute the close command and close the collector. __del__ is automatically called to \
                destroy the collector instance when the collector finishes its work
        """
        self.close()

    """
    MuZero related method
    """

    def get_priorities(self, i, pred_values_lst, search_values_lst):
        """
        Overview:
            obtain the priorities at index i.
        """
        if self.game_config.use_priority and not self.game_config.use_max_priority:
            pred_values = torch.from_numpy(np.array(pred_values_lst[i])).to(self.game_config.device).float().view(-1)
            search_values = torch.from_numpy(np.array(search_values_lst[i])).to(self.game_config.device).float().view(-1)
            priorities = L1Loss(reduction='none'
                                )(pred_values,
                                  search_values).detach().cpu().numpy() + self.game_config.prioritized_replay_eps
        else:
            # priorities is None -> use the max priority for all newly collected data
            priorities = None

        return priorities

    def len_pool(self):
        """
        Overview:
            current pool size
        """
        return len(self.trajectory_pool)

    def free(self, end_tag):
        """
        Overview:
            save the game histories and clear the pool
            self.trajectory_pool: list of (game_history, priority)
        """
        if self.len_pool() >= self.pool_size:
            self.replay_buffer.push_games(
                [self.trajectory_pool[i][0] for i in range(self.len_pool())], [
                    {
                        'end_tag': end_tag,
                        'priorities': self.trajectory_pool[i][1],
                        'gap_steps': self.gap_step
                    } for i in range(self.len_pool())
                ]
            )

            del self.trajectory_pool[:]

    def put_last_trajectory(self, i, last_game_histories, last_game_priorities, game_histories, done):
        """
        Overview:
            put the last game history into the pool if the current game is finished
        Arguments:
            - last_game_histories (:obj:`list`): list of the last game histories
            - last_game_priorities (:obj:`list`): list of the last game priorities
            - game_histories (:obj:`list`): list of the current game histories
        Note:
            (last_game_histories[i].obs_history[-4:] == game_histories[i].obs_history[:4]) is True
        """
        # pad over last block trajectory
        beg_index = self.game_config.stacked_observations
        end_index = beg_index + self.game_config.num_unroll_steps

        # the start 4 obs is init zero obs, so we take the 4th -(4+5）th obs as the pad obs
        pad_obs_lst = game_histories[i].obs_history[beg_index:end_index]
        pad_child_visits_lst = game_histories[i].child_visits[beg_index:end_index]

        beg_index = 0
        # self.gap_step = self.game_config.num_unroll_steps + self.game_config.td_steps
        end_index = beg_index + self.gap_step - 1

        pad_reward_lst = game_histories[i].rewards[beg_index:end_index]

        beg_index = 0
        end_index = beg_index + self.gap_step

        pad_root_values_lst = game_histories[i].root_values[beg_index:end_index]

        # pad over and save
        last_game_histories[i].pad_over(pad_obs_lst, pad_reward_lst, pad_root_values_lst, pad_child_visits_lst)

        """
        Note:
            game_history element shape:
            obs: game_history_length + stack + num_unroll_steps, 20+4 +5  
            rew: game_history_length + stack + num_unroll_steps + td_steps -1  20 +5+3-1  
            action: game_history_length -> 20  
            root_values:  game_history_length + num_unroll_steps + td_steps -> 20 +5+3
            child_visits： game_history_length + num_unroll_steps -> 20 +5
            to_play: game_history_length -> 20  
            action_mask: game_history_length -> 20 
        """

        last_game_histories[i].game_over()

        # put the game history into the pool
        # try:
        #     assert len(last_game_histories[i]) == len(last_game_priorities[i])
        # except Exception as error:
        #     print(error)
        self.trajectory_pool.append((last_game_histories[i], last_game_priorities[i], done[i]))
        # self.free(done[i])

        # reset last game_histories
        last_game_histories[i] = None
        last_game_priorities[i] = None

        return None

    def collect(self,
                n_episode: Optional[int] = None,
                train_iter: int = 0,
                policy_kwargs: Optional[dict] = None) -> List[Any]:
        """
        Overview:
            Collect `n_episode` data with policy_kwargs, which is already trained `train_iter` iterations
        Arguments:
            - n_episode (:obj:`int`): the number of collecting data episode
            - train_iter (:obj:`int`): the number of training iteration
            - policy_kwargs (:obj:`dict`): the keyword args for policy forward
        Returns:
            - return_data (:obj:`List`): A list containing collected episodes if not get_train_sample, otherwise, \
                return train_samples split by unroll_len.
        """
        if n_episode is None:
            if self._default_n_episode is None:
                raise RuntimeError("Please specify collect n_episode")
            else:
                n_episode = self._default_n_episode
        assert n_episode >= self._env_num, "Please make sure n_episode >= env_num{}/{}".format(n_episode, self._env_num)
        if policy_kwargs is None:
            policy_kwargs = {}
        temperature = policy_kwargs['temperature']
        collected_episode = 0
        return_data = []
        env_nums = self._env_num
        # initializations
        init_obses = self._env.ready_obs
        action_mask = [to_ndarray(init_obses[i]['action_mask']) for i in range(env_nums)]
        if 'to_play' in init_obses[0]:
            two_plaer_game = True
        else:
            two_plaer_game = False

        if two_plaer_game:
            to_play = [to_ndarray(init_obses[i]['to_play']) for i in range(env_nums)]

        dones = np.array([False for _ in range(env_nums)])
        game_histories = [
            GameHistory(
                self._env.action_space, max_length=self.game_config.game_history_max_length, config=self.game_config
            ) for _ in range(env_nums)
        ]
        for i in range(env_nums):
            game_histories[i].init(
                [to_ndarray(init_obses[i]['observation']) for _ in range(self.game_config.stacked_observations)]
            )

        last_game_histories = [None for _ in range(env_nums)]
        last_game_priorities = [None for _ in range(env_nums)]

        # stack observation windows in boundary: s398, s399, s400, current s1 -> for not init trajectory
        stack_obs_windows = [[] for _ in range(env_nums)]
        for i in range(env_nums):
            stack_obs_windows[i] = [
                to_ndarray(init_obses[i]['observation']) for _ in range(self.game_config.stacked_observations)
            ]
            game_histories[i].init(stack_obs_windows[i])

        # for priorities in self-play
        search_values_lst = [[] for _ in range(env_nums)]
        pred_values_lst = [[] for _ in range(env_nums)]

        # some logs
        eps_ori_reward_lst, eps_reward_lst, eps_steps_lst, visit_entropies_lst = np.zeros(env_nums), np.zeros(
            env_nums
        ), np.zeros(env_nums), np.zeros(env_nums)

        self_play_rewards = 0.
        self_play_ori_rewards = 0.
        self_play_moves = 0.
        self_play_episodes = 0.

        self_play_rewards_max = -np.inf
        self_play_moves_max = 0

        self_play_visit_entropy = []
        total_transitions = 0

        def _get_max_entropy(action_space):
            p = 1.0 / action_space
            ep = -action_space * p * np.log2(p)
            return ep

        max_visit_entropy = _get_max_entropy(self.game_config.action_space_size)
        print('max_visit_entropy', max_visit_entropy)

        while True:
            with self._timer:
                stack_obs = [game_history.step_obs() for game_history in game_histories]
                stack_obs = to_ndarray(stack_obs)
                stack_obs = prepare_observation_lst(stack_obs)
                if self.game_config.image_based:
                    stack_obs = torch.from_numpy(stack_obs).to(self.game_config.device).float() / 255.0
                else:
                    stack_obs = torch.from_numpy(np.array(stack_obs)).to(self.game_config.device)

                if two_plaer_game:
                    policy_output = self._policy.forward(stack_obs, action_mask, temperature, to_play)
                else:
                    policy_output = self._policy.forward(stack_obs, action_mask, temperature, None)

                actions = {k: v['action'] for k, v in policy_output.items()}
                distributions_dict = {k: v['distributions'] for k, v in policy_output.items()}
                value_dict = {k: v['value'] for k, v in policy_output.items()}
                pred_value_dict = {k: v['pred_value'] for k, v in policy_output.items()}
                visit_entropy_dict = {k: v['visit_entropy'] for k, v in policy_output.items()}

                timesteps = self._env.step(actions)

                # if two_plaer_game:
                #     action_mask = [to_ndarray(timesteps[i].obs['action_mask']) for i in range(env_nums)]
                #     to_play = [to_ndarray(timesteps[i].obs['to_play']) for i in range(env_nums)]

            # TODO(nyz) this duration may be inaccurate in async env
            interaction_duration = self._timer.value / len(timesteps)

            for env_id, timestep in timesteps.items():
                i = env_id
                obs, ori_reward, done, info = timestep.obs, timestep.reward, timestep.done, timestep.info

                if self.game_config.clip_reward:
                    clip_reward = np.sign(ori_reward)
                else:
                    clip_reward = ori_reward
                game_histories[i].store_search_stats(distributions_dict[i], value_dict[i])
                if two_plaer_game:
                    # for two_player board games
                    game_histories[i].append(
                        actions[i], to_ndarray(obs['observation']), clip_reward, action_mask[i], to_play[i]
                    )
                else:
                    game_histories[i].append(actions[i], to_ndarray(obs['observation']), clip_reward)

                # NOTE: the position is very important
                if two_plaer_game:
                    action_mask[i] = to_ndarray(obs['action_mask'])
                    to_play[i] = to_ndarray(timesteps[i].obs['to_play'])

                eps_reward_lst[i] += clip_reward
                eps_ori_reward_lst[i] += ori_reward
                dones[i] = done
                visit_entropies_lst[i] += visit_entropy_dict[i]

                eps_steps_lst[i] += 1
                total_transitions += 1

                if self.game_config.use_priority and not self.game_config.use_max_priority:
                    pred_values_lst[i].append(pred_value_dict[i])
                    search_values_lst[i].append(value_dict[i])

                # fresh stack windows
                del stack_obs_windows[i][0]
                stack_obs_windows[i].append(to_ndarray(obs['observation']))

                # we will save a game history if it is the end of the game or the next game history is finished.

                # if game history is full, we will save the game history
                if game_histories[i].is_full():
                    # pad over last block trajectory
                    if last_game_histories[i] is not None:
                        # TODO(pu): return the one game history
                        self.put_last_trajectory(
                            i, last_game_histories, last_game_priorities, game_histories, dones
                        )

                    # calculate priority
                    priorities = self.get_priorities(i, pred_values_lst, search_values_lst)
                    pred_values_lst[i] = []
                    search_values_lst[i] = []

                    #  the game_histories become last_game_history
                    last_game_histories[i] = game_histories[i]
                    last_game_priorities[i] = priorities

                    # new GameHistory
                    game_histories[i] = GameHistory(
                        self._env.action_space,
                        max_length=self.game_config.game_history_max_length,
                        config=self.game_config
                    )
                    game_histories[i].init(stack_obs_windows[i])
                    # TODO(pu): return data

                self._env_info[env_id]['step'] += 1
                self._total_envstep_count += 1
                self._env_info[env_id]['time'] += self._timer.value + interaction_duration

                if timestep.done:
                    self._total_episode_count += 1
                    reward = timestep.info['final_eval_reward']
                    info = {
                        'reward': reward,
                        'time': self._env_info[env_id]['time'],
                        'step': self._env_info[env_id]['step'],
                    }
                    collected_episode += 1
                    self._episode_info.append(info)
                    # Env reset is done by env_manager automatically
                    self._policy.reset([env_id])
                    self._reset_stat(env_id)

                    # if it is the end of the game, we will save the game history

                    # NOTE: put the  second last game history in one episode into the trajectory_pool
                    # pad over 2th last game_history using the last game_history
                    if last_game_histories[i] is not None:
                        self.put_last_trajectory(i, last_game_histories, last_game_priorities, game_histories, dones)

                    # store current block trajectory
                    priorities = self.get_priorities(i, pred_values_lst, search_values_lst)

                    # NOTE: put the last game history in one episode into the trajectory_pool
                    game_histories[i].game_over()

                    # assert len(game_histories[i]) == len(priorities)
                    self.trajectory_pool.append((game_histories[i], priorities, dones[i]))
                    # NOTE: this is very important to save the done data to replay_buffer
                    # self.free(end_tag=True)

                    # reset the finished env and init game_histories
                    init_obses = self._env.ready_obs
                    init_obs = init_obses[i]['observation']
                    init_obs = to_ndarray(init_obs)
                    action_mask[i] = to_ndarray(init_obses[i]['action_mask'])
                    to_play[i] = to_ndarray(init_obses[i]['to_play'])

                    game_histories[i] = GameHistory(
                        self._env.action_space,
                        max_length=self.game_config.game_history_max_length,
                        config=self.game_config
                    )
                    stack_obs_windows[i] = [init_obs for _ in range(self.game_config.stacked_observations)]
                    game_histories[i].init(stack_obs_windows[i])
                    last_game_histories[i] = None
                    last_game_priorities[i] = None

                    # log
                    self_play_rewards_max = max(self_play_rewards_max, eps_reward_lst[i])
                    self_play_moves_max = max(self_play_moves_max, eps_steps_lst[i])
                    self_play_rewards += eps_reward_lst[i]
                    self_play_ori_rewards += eps_ori_reward_lst[i]
                    self_play_visit_entropy.append(visit_entropies_lst[i] / eps_steps_lst[i])
                    self_play_moves += eps_steps_lst[i]
                    self_play_episodes += 1

                    pred_values_lst[i] = []
                    search_values_lst[i] = []
                    eps_steps_lst[i] = 0
                    eps_reward_lst[i] = 0
                    eps_ori_reward_lst[i] = 0
                    visit_entropies_lst[i] = 0

            if collected_episode >= n_episode:
                # logs
                visit_entropies = np.array(self_play_visit_entropy).mean()
                # visit_entropies /= max_visit_entropy
                print('visit_entropies:', visit_entropies)

                # save the game histories and clear the pool
                # self.trajectory_pool: list of (game_history, priority)
                self.replay_buffer.push_games(
                    [self.trajectory_pool[i][0] for i in range(self.len_pool())], [
                        {
                            'priorities': self.trajectory_pool[i][1],
                            'end_tag': self.trajectory_pool[i][2],
                            'gap_steps': self.gap_step
                        } for i in range(self.len_pool())
                    ]
                )
                # np.save('/Users/puyuan/code/DI-engine/dizoo/board_games/atari/config/one_episode_replay_buffer_img',
                #         self.replay_buffer.buffer)
                # one_episode_replay_buffer_img = np.load('/Users/puyuan/code/DI-engine/dizoo/board_games/atari/config/one_episode_replay_buffer_img.npy',
                #         allow_pickle=True)
                del self.trajectory_pool[:]
                break
        # log
        self._output_log(train_iter)
        return return_data

    def _output_log(self, train_iter: int) -> None:
        """
        Overview:
            Print the output log information. You can refer to Docs/Best Practice/How to understand\
             training generated folders/Serial mode/log/collector for more details.
        Arguments:
            - train_iter (:obj:`int`): the number of training iteration.
        """
        if (train_iter - self._last_train_iter) >= self._collect_print_freq and len(self._episode_info) > 0:
            self._last_train_iter = train_iter
            episode_count = len(self._episode_info)
            envstep_count = sum([d['step'] for d in self._episode_info])
            duration = sum([d['time'] for d in self._episode_info])
            episode_reward = [d['reward'] for d in self._episode_info]
            self._total_duration += duration
            info = {
                'episode_count': episode_count,
                'envstep_count': envstep_count,
                'avg_envstep_per_episode': envstep_count / episode_count,
                'avg_envstep_per_sec': envstep_count / duration,
                'avg_episode_per_sec': episode_count / duration,
                'collect_time': duration,
                'reward_mean': np.mean(episode_reward),
                'reward_std': np.std(episode_reward),
                'reward_max': np.max(episode_reward),
                'reward_min': np.min(episode_reward),
                'total_envstep_count': self._total_envstep_count,
                'total_episode_count': self._total_episode_count,
                'total_duration': self._total_duration,
                # 'each_reward': episode_reward,
            }
            self._episode_info.clear()
            self._logger.info("collect end:\n{}".format('\n'.join(['{}: {}'.format(k, v) for k, v in info.items()])))
            for k, v in info.items():
                if k in ['each_reward']:
                    continue
                self._tb_logger.add_scalar('{}_iter/'.format(self._instance_name) + k, v, train_iter)
                if k in ['total_envstep_count']:
                    continue
                self._tb_logger.add_scalar('{}_step/'.format(self._instance_name) + k, v, self._total_envstep_count)
