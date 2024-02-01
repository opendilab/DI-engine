from typing import List, Dict, Tuple
from ditk import logging
from copy import deepcopy
from easydict import EasyDict
from torch.utils.data import Dataset
from dataclasses import dataclass

import pickle
import easydict
import torch
import numpy as np

from ding.utils.bfs_helper import get_vi_sequence
from ding.utils import DATASET_REGISTRY, import_module, DatasetNormalizer
from ding.rl_utils import discount_cumsum


@dataclass
class DatasetStatistics:
    """
    Overview:
        Dataset statistics.
    """
    mean: np.ndarray  # obs
    std: np.ndarray  # obs
    action_bounds: np.ndarray


@DATASET_REGISTRY.register('naive')
class NaiveRLDataset(Dataset):
    """
    Overview:
        Naive RL dataset, which is used for offline RL algorithms.
    Interfaces:
        ``__init__``, ``__len__``, ``__getitem__``
    """

    def __init__(self, cfg) -> None:
        """
        Overview:
            Initialization method.
        Arguments:
            - cfg (:obj:`dict`): Config dict.
        """

        assert type(cfg) in [str, EasyDict], "invalid cfg type: {}".format(type(cfg))
        if isinstance(cfg, EasyDict):
            self._data_path = cfg.policy.collect.data_path
        elif isinstance(cfg, str):
            self._data_path = cfg
        with open(self._data_path, 'rb') as f:
            self._data: List[Dict[str, torch.Tensor]] = pickle.load(f)

    def __len__(self) -> int:
        """
        Overview:
            Get the length of the dataset.
        """

        return len(self._data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Overview:
            Get the item of the dataset.
        """

        return self._data[idx]


@DATASET_REGISTRY.register('d4rl')
class D4RLDataset(Dataset):
    """
    Overview:
        D4RL dataset, which is used for offline RL algorithms.
    Interfaces:
        ``__init__``, ``__len__``, ``__getitem__``
    Properties:
        - mean (:obj:`np.ndarray`): Mean of the dataset.
        - std (:obj:`np.ndarray`): Std of the dataset.
        - action_bounds (:obj:`np.ndarray`): Action bounds of the dataset.
        - statistics (:obj:`dict`): Statistics of the dataset.
    """

    def __init__(self, cfg: dict) -> None:
        """
        Overview:
            Initialization method.
        Arguments:
            - cfg (:obj:`dict`): Config dict.
        """

        import gym
        try:
            import d4rl  # register d4rl enviroments with open ai gym
        except ImportError:
            import sys
            logging.warning("not found d4rl env, please install it, refer to https://github.com/rail-berkeley/d4rl")
            sys.exit(1)

        # Init parameters
        data_path = cfg.policy.collect.get('data_path', None)
        env_id = cfg.env.env_id

        # Create the environment
        if data_path:
            d4rl.set_dataset_path(data_path)
        env = gym.make(env_id)
        dataset = d4rl.qlearning_dataset(env)
        self._cal_statistics(dataset, env)
        try:
            if cfg.env.norm_obs.use_norm and cfg.env.norm_obs.offline_stats.use_offline_stats:
                dataset = self._normalize_states(dataset)
        except (KeyError, AttributeError):
            # do not normalize
            pass
        self._data = []
        self._load_d4rl(dataset)

    @property
    def data(self) -> List:
        return self._data

    def __len__(self) -> int:
        """
        Overview:
            Get the length of the dataset.
        """

        return len(self._data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Overview:
            Get the item of the dataset.
        """

        return self._data[idx]

    def _load_d4rl(self, dataset: Dict[str, np.ndarray]) -> None:
        """
        Overview:
            Load the d4rl dataset.
        Arguments:
            - dataset (:obj:`Dict[str, np.ndarray]`): The d4rl dataset.
        """

        for i in range(len(dataset['observations'])):
            trans_data = {}
            trans_data['obs'] = torch.from_numpy(dataset['observations'][i])
            trans_data['next_obs'] = torch.from_numpy(dataset['next_observations'][i])
            trans_data['action'] = torch.from_numpy(dataset['actions'][i])
            trans_data['reward'] = torch.tensor(dataset['rewards'][i])
            trans_data['done'] = dataset['terminals'][i]
            self._data.append(trans_data)

    def _cal_statistics(self, dataset, env, eps=1e-3, add_action_buffer=True):
        """
        Overview:
            Calculate the statistics of the dataset.
        Arguments:
            - dataset (:obj:`Dict[str, np.ndarray]`): The d4rl dataset.
            - env (:obj:`gym.Env`): The environment.
            - eps (:obj:`float`): Epsilon.
        """

        self._mean = dataset['observations'].mean(0)
        self._std = dataset['observations'].std(0) + eps
        action_max = dataset['actions'].max(0)
        action_min = dataset['actions'].min(0)
        if add_action_buffer:
            action_buffer = 0.05 * (action_max - action_min)
            action_max = (action_max + action_buffer).clip(max=env.action_space.high)
            action_min = (action_min - action_buffer).clip(min=env.action_space.low)
        self._action_bounds = np.stack([action_min, action_max], axis=0)

    def _normalize_states(self, dataset):
        """
        Overview:
            Normalize the states.
        Arguments:
            - dataset (:obj:`Dict[str, np.ndarray]`): The d4rl dataset.
        """

        dataset['observations'] = (dataset['observations'] - self._mean) / self._std
        dataset['next_observations'] = (dataset['next_observations'] - self._mean) / self._std
        return dataset

    @property
    def mean(self):
        """
        Overview:
            Get the mean of the dataset.
        """

        return self._mean

    @property
    def std(self):
        """
        Overview:
            Get the std of the dataset.
        """

        return self._std

    @property
    def action_bounds(self) -> np.ndarray:
        """
        Overview:
            Get the action bounds of the dataset.
        """

        return self._action_bounds

    @property
    def statistics(self) -> dict:
        """
        Overview:
            Get the statistics of the dataset.
        """

        return DatasetStatistics(mean=self.mean, std=self.std, action_bounds=self.action_bounds)


@DATASET_REGISTRY.register('hdf5')
class HDF5Dataset(Dataset):
    """
    Overview:
        HDF5 dataset is saved in hdf5 format, which is used for offline RL algorithms.
        The hdf5 format is a common format for storing large numerical arrays in Python.
        For more details, please refer to https://support.hdfgroup.org/HDF5/.
    Interfaces:
        ``__init__``, ``__len__``, ``__getitem__``
    Properties:
        - mean (:obj:`np.ndarray`): Mean of the dataset.
        - std (:obj:`np.ndarray`): Std of the dataset.
        - action_bounds (:obj:`np.ndarray`): Action bounds of the dataset.
        - statistics (:obj:`dict`): Statistics of the dataset.
    """

    def __init__(self, cfg: dict) -> None:
        """
        Overview:
            Initialization method.
        Arguments:
            - cfg (:obj:`dict`): Config dict.
        """

        try:
            import h5py
        except ImportError:
            import sys
            logging.warning("not found h5py package, please install it trough `pip install h5py ")
            sys.exit(1)
        data_path = cfg.policy.collect.get('data_path', None)
        if 'dataset' in cfg:
            self.context_len = cfg.dataset.context_len
        else:
            self.context_len = 0
        data = h5py.File(data_path, 'r')
        self._load_data(data)
        self._cal_statistics()
        try:
            if cfg.env.norm_obs.use_norm and cfg.env.norm_obs.offline_stats.use_offline_stats:
                self._normalize_states()
        except (KeyError, AttributeError):
            # do not normalize
            pass

    def __len__(self) -> int:
        """
        Overview:
            Get the length of the dataset.
        """

        return len(self._data['obs']) - self.context_len

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Overview:
            Get the item of the dataset.
        Arguments:
            - idx (:obj:`int`): The index of the dataset.
        """

        if self.context_len == 0:  # for other offline RL algorithms
            return {k: self._data[k][idx] for k in self._data.keys()}
        else:  # for decision transformer
            block_size = self.context_len
            done_idx = idx + block_size
            idx = done_idx - block_size
            states = torch.as_tensor(
                np.array(self._data['obs'][idx:done_idx]), dtype=torch.float32
            ).view(block_size, -1)
            actions = torch.as_tensor(self._data['action'][idx:done_idx], dtype=torch.long)
            rtgs = torch.as_tensor(self._data['reward'][idx:done_idx, 0], dtype=torch.float32)
            timesteps = torch.as_tensor(range(idx, done_idx), dtype=torch.int64)
            traj_mask = torch.ones(self.context_len, dtype=torch.long)
            return timesteps, states, actions, rtgs, traj_mask

    def _load_data(self, dataset: Dict[str, np.ndarray]) -> None:
        """
        Overview:
            Load the dataset.
        Arguments:
            - dataset (:obj:`Dict[str, np.ndarray]`): The dataset.
        """

        self._data = {}
        for k in dataset.keys():
            logging.info(f'Load {k} data.')
            self._data[k] = dataset[k][:]

    def _cal_statistics(self, eps: float = 1e-3):
        """
        Overview:
            Calculate the statistics of the dataset.
        Arguments:
            - eps (:obj:`float`): Epsilon.
        """

        self._mean = self._data['obs'].mean(0)
        self._std = self._data['obs'].std(0) + eps
        action_max = self._data['action'].max(0)
        action_min = self._data['action'].min(0)
        buffer = 0.05 * (action_max - action_min)
        action_max = action_max.astype(float) + buffer
        action_min = action_max.astype(float) - buffer
        self._action_bounds = np.stack([action_min, action_max], axis=0)

    def _normalize_states(self):
        """
        Overview:
            Normalize the states.
        """

        self._data['obs'] = (self._data['obs'] - self._mean) / self._std
        self._data['next_obs'] = (self._data['next_obs'] - self._mean) / self._std

    @property
    def mean(self):
        """
        Overview:
            Get the mean of the dataset.
        """

        return self._mean

    @property
    def std(self):
        """
        Overview:
            Get the std of the dataset.
        """

        return self._std

    @property
    def action_bounds(self) -> np.ndarray:
        """
        Overview:
            Get the action bounds of the dataset.
        """

        return self._action_bounds

    @property
    def statistics(self) -> dict:
        """
        Overview:
            Get the statistics of the dataset.
        """

        return DatasetStatistics(mean=self.mean, std=self.std, action_bounds=self.action_bounds)


@DATASET_REGISTRY.register('d4rl_trajectory')
class D4RLTrajectoryDataset(Dataset):
    """
    Overview:
        D4RL trajectory dataset, which is used for offline RL algorithms.
    Interfaces:
        ``__init__``, ``__len__``, ``__getitem__``
    """

    # from infos.py from official d4rl github repo
    REF_MIN_SCORE = {
        'halfcheetah': -280.178953,
        'walker2d': 1.629008,
        'hopper': -20.272305,
    }

    REF_MAX_SCORE = {
        'halfcheetah': 12135.0,
        'walker2d': 4592.3,
        'hopper': 3234.3,
    }

    # calculated from d4rl datasets
    D4RL_DATASET_STATS = {
        'halfcheetah-medium-v2': {
            'state_mean': [
                -0.06845773756504059, 0.016414547339081764, -0.18354906141757965, -0.2762460708618164,
                -0.34061527252197266, -0.09339715540409088, -0.21321271359920502, -0.0877423882484436,
                5.173007488250732, -0.04275195300579071, -0.036108363419771194, 0.14053793251514435,
                0.060498327016830444, 0.09550975263118744, 0.06739100068807602, 0.005627387668937445,
                0.013382787816226482
            ],
            'state_std': [
                0.07472999393939972, 0.3023499846458435, 0.30207309126853943, 0.34417077898979187, 0.17619241774082184,
                0.507205605506897, 0.2567007839679718, 0.3294812738895416, 1.2574149370193481, 0.7600541710853577,
                1.9800915718078613, 6.565362453460693, 7.466367721557617, 4.472222805023193, 10.566964149475098,
                5.671932697296143, 7.4982590675354
            ]
        },
        'halfcheetah-medium-replay-v2': {
            'state_mean': [
                -0.12880703806877136, 0.3738119602203369, -0.14995987713336945, -0.23479078710079193,
                -0.2841278612613678, -0.13096535205841064, -0.20157982409000397, -0.06517726927995682,
                3.4768247604370117, -0.02785065770149231, -0.015035249292850494, 0.07697279006242752,
                0.01266712136566639, 0.027325302362442017, 0.02316424623131752, 0.010438721626996994,
                -0.015839405357837677
            ],
            'state_std': [
                0.17019015550613403, 1.284424901008606, 0.33442774415016174, 0.3672759234905243, 0.26092398166656494,
                0.4784106910228729, 0.3181420564651489, 0.33552637696266174, 2.0931615829467773, 0.8037433624267578,
                1.9044333696365356, 6.573209762573242, 7.572863578796387, 5.069749355316162, 9.10555362701416,
                6.085654258728027, 7.25300407409668
            ]
        },
        'halfcheetah-medium-expert-v2': {
            'state_mean': [
                -0.05667462572455406, 0.024369969964027405, -0.061670560389757156, -0.22351515293121338,
                -0.2675151228904724, -0.07545716315507889, -0.05809682980179787, -0.027675075456500053,
                8.110626220703125, -0.06136331334710121, -0.17986927926540375, 0.25175222754478455, 0.24186332523822784,
                0.2519369423389435, 0.5879552960395813, -0.24090635776519775, -0.030184272676706314
            ],
            'state_std': [
                0.06103534251451492, 0.36054104566574097, 0.45544400811195374, 0.38476887345314026, 0.2218363732099533,
                0.5667523741722107, 0.3196682929992676, 0.2852923572063446, 3.443821907043457, 0.6728139519691467,
                1.8616976737976074, 9.575807571411133, 10.029894828796387, 5.903450012207031, 12.128185272216797,
                6.4811787605285645, 6.378620147705078
            ]
        },
        'walker2d-medium-v2': {
            'state_mean': [
                1.218966007232666, 0.14163373410701752, -0.03704913705587387, -0.13814310729503632, 0.5138224363327026,
                -0.04719110205769539, -0.47288352251052856, 0.042254164814949036, 2.3948874473571777,
                -0.03143199160695076, 0.04466355964541435, -0.023907244205474854, -0.1013401448726654,
                0.09090937674045563, -0.004192637279629707, -0.12120571732521057, -0.5497063994407654
            ],
            'state_std': [
                0.12311358004808426, 0.3241879940032959, 0.11456084251403809, 0.2623065710067749, 0.5640279054641724,
                0.2271878570318222, 0.3837319612503052, 0.7373676896095276, 1.2387926578521729, 0.798020601272583,
                1.5664079189300537, 1.8092705011367798, 3.025604248046875, 4.062486171722412, 1.4586567878723145,
                3.7445690631866455, 5.5851287841796875
            ]
        },
        'walker2d-medium-replay-v2': {
            'state_mean': [
                1.209364652633667, 0.13264022767543793, -0.14371201395988464, -0.2046516090631485, 0.5577612519264221,
                -0.03231537342071533, -0.2784661054611206, 0.19130706787109375, 1.4701707363128662,
                -0.12504704296588898, 0.0564953051507473, -0.09991033375263214, -0.340340256690979, 0.03546293452382088,
                -0.08934258669614792, -0.2992438077926636, -0.5984178185462952
            ],
            'state_std': [
                0.11929835379123688, 0.3562574088573456, 0.25852200388908386, 0.42075422406196594, 0.5202291011810303,
                0.15685082972049713, 0.36770978569984436, 0.7161387801170349, 1.3763766288757324, 0.8632221817970276,
                2.6364643573760986, 3.0134117603302, 3.720684051513672, 4.867283821105957, 2.6681625843048096,
                3.845186948776245, 5.4768385887146
            ]
        },
        'walker2d-medium-expert-v2': {
            'state_mean': [
                1.2294334173202515, 0.16869689524173737, -0.07089081406593323, -0.16197483241558075,
                0.37101927399635315, -0.012209027074277401, -0.42461398243904114, 0.18986578285694122,
                3.162475109100342, -0.018092676997184753, 0.03496946766972542, -0.013921679928898811,
                -0.05937029421329498, -0.19549426436424255, -0.0019200450042262673, -0.062483321875333786,
                -0.27366524934768677
            ],
            'state_std': [
                0.09932824969291687, 0.25981399416923523, 0.15062759816646576, 0.24249176681041718, 0.6758718490600586,
                0.1650741547346115, 0.38140663504600525, 0.6962361335754395, 1.3501490354537964, 0.7641991376876831,
                1.534574270248413, 2.1785972118377686, 3.276582717895508, 4.766193866729736, 1.1716983318328857,
                4.039782524108887, 5.891613960266113
            ]
        },
        'hopper-medium-v2': {
            'state_mean': [
                1.311279058456421, -0.08469521254301071, -0.5382719039916992, -0.07201576232910156, 0.04932365566492081,
                2.1066856384277344, -0.15017354488372803, 0.008783451281487942, -0.2848185896873474,
                -0.18540096282958984, -0.28461286425590515
            ],
            'state_std': [
                0.17790751159191132, 0.05444620922207832, 0.21297138929367065, 0.14530418813228607, 0.6124444007873535,
                0.8517446517944336, 1.4515252113342285, 0.6751695871353149, 1.5362390279769897, 1.616074562072754,
                5.607253551483154
            ]
        },
        'hopper-medium-replay-v2': {
            'state_mean': [
                1.2305138111114502, -0.04371410980820656, -0.44542956352233887, -0.09370097517967224,
                0.09094487875699997, 1.3694725036621094, -0.19992674887180328, -0.022861352190375328,
                -0.5287045240402222, -0.14465883374214172, -0.19652697443962097
            ],
            'state_std': [
                0.1756512075662613, 0.0636928603053093, 0.3438323438167572, 0.19566889107227325, 0.5547984838485718,
                1.051029920578003, 1.158307671546936, 0.7963128685951233, 1.4802359342575073, 1.6540331840515137,
                5.108601093292236
            ]
        },
        'hopper-medium-expert-v2': {
            'state_mean': [
                1.3293815851211548, -0.09836531430482864, -0.5444297790527344, -0.10201650857925415,
                0.02277466468513012, 2.3577215671539307, -0.06349576264619827, -0.00374026270583272,
                -0.1766270101070404, -0.11862941086292267, -0.12097819894552231
            ],
            'state_std': [
                0.17012375593185425, 0.05159067362546921, 0.18141433596611023, 0.16430604457855225, 0.6023368239402771,
                0.7737284898757935, 1.4986555576324463, 0.7483318448066711, 1.7953159809112549, 2.0530025959014893,
                5.725032806396484
            ]
        },
    }

    def __init__(self, cfg: dict) -> None:
        """
        Overview:
            Initialization method.
        Arguments:
            - cfg (:obj:`dict`): Config dict.
        """

        dataset_path = cfg.dataset.data_dir_prefix
        rtg_scale = cfg.dataset.rtg_scale
        self.context_len = cfg.dataset.context_len
        self.env_type = cfg.dataset.env_type

        if 'hdf5' in dataset_path:  # for mujoco env
            try:
                import h5py
                import collections
            except ImportError:
                import sys
                logging.warning("not found h5py package, please install it trough `pip install h5py ")
                sys.exit(1)
            dataset = h5py.File(dataset_path, 'r')

            N = dataset['rewards'].shape[0]
            data_ = collections.defaultdict(list)

            use_timeouts = False
            if 'timeouts' in dataset:
                use_timeouts = True

            episode_step = 0
            paths = []
            for i in range(N):
                done_bool = bool(dataset['terminals'][i])
                if use_timeouts:
                    final_timestep = dataset['timeouts'][i]
                else:
                    final_timestep = (episode_step == 1000 - 1)
                for k in ['observations', 'actions', 'rewards', 'terminals']:
                    data_[k].append(dataset[k][i])
                if done_bool or final_timestep:
                    episode_step = 0
                    episode_data = {}
                    for k in data_:
                        episode_data[k] = np.array(data_[k])
                    paths.append(episode_data)
                    data_ = collections.defaultdict(list)
                episode_step += 1

            self.trajectories = paths

            # calculate state mean and variance and returns_to_go for all traj
            states = []
            for traj in self.trajectories:
                traj_len = traj['observations'].shape[0]
                states.append(traj['observations'])
                # calculate returns to go and rescale them
                traj['returns_to_go'] = discount_cumsum(traj['rewards'], 1.0) / rtg_scale

            # used for input normalization
            states = np.concatenate(states, axis=0)
            self.state_mean, self.state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6

            # normalize states
            for traj in self.trajectories:
                traj['observations'] = (traj['observations'] - self.state_mean) / self.state_std

        elif 'pkl' in dataset_path:
            if 'dqn' in dataset_path:
                # load dataset
                with open(dataset_path, 'rb') as f:
                    self.trajectories = pickle.load(f)

                if isinstance(self.trajectories[0], list):
                    # for our collected dataset, e.g. cartpole/lunarlander case
                    trajectories_tmp = []

                    original_keys = ['obs', 'next_obs', 'action', 'reward']
                    keys = ['observations', 'next_observations', 'actions', 'rewards']
                    trajectories_tmp = [
                        {
                            key: np.stack(
                                [
                                    self.trajectories[eps_index][transition_index][o_key]
                                    for transition_index in range(len(self.trajectories[eps_index]))
                                ],
                                axis=0
                            )
                            for key, o_key in zip(keys, original_keys)
                        } for eps_index in range(len(self.trajectories))
                    ]
                    self.trajectories = trajectories_tmp

                states = []
                for traj in self.trajectories:
                    # traj_len = traj['observations'].shape[0]
                    states.append(traj['observations'])
                    # calculate returns to go and rescale them
                    traj['returns_to_go'] = discount_cumsum(traj['rewards'], 1.0) / rtg_scale

                # used for input normalization
                states = np.concatenate(states, axis=0)
                self.state_mean, self.state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6

                # normalize states
                for traj in self.trajectories:
                    traj['observations'] = (traj['observations'] - self.state_mean) / self.state_std
            else:
                # load dataset
                with open(dataset_path, 'rb') as f:
                    self.trajectories = pickle.load(f)

                states = []
                for traj in self.trajectories:
                    states.append(traj['observations'])
                    # calculate returns to go and rescale them
                    traj['returns_to_go'] = discount_cumsum(traj['rewards'], 1.0) / rtg_scale

                # used for input normalization
                states = np.concatenate(states, axis=0)
                self.state_mean, self.state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6

                # normalize states
                for traj in self.trajectories:
                    traj['observations'] = (traj['observations'] - self.state_mean) / self.state_std
        else:
            # -- load data from memory (make more efficient)
            obss = []
            actions = []
            returns = [0]
            done_idxs = []
            stepwise_returns = []

            transitions_per_buffer = np.zeros(50, dtype=int)
            num_trajectories = 0
            while len(obss) < cfg.dataset.num_steps:
                buffer_num = np.random.choice(np.arange(50 - cfg.dataset.num_buffers, 50), 1)[0]
                i = transitions_per_buffer[buffer_num]
                frb = FixedReplayBuffer(
                    data_dir=cfg.dataset.data_dir_prefix + '/1/replay_logs',
                    replay_suffix=buffer_num,
                    observation_shape=(84, 84),
                    stack_size=4,
                    update_horizon=1,
                    gamma=0.99,
                    observation_dtype=np.uint8,
                    batch_size=32,
                    replay_capacity=100000
                )
                if frb._loaded_buffers:
                    done = False
                    curr_num_transitions = len(obss)
                    trajectories_to_load = cfg.dataset.trajectories_per_buffer
                    while not done:
                        states, ac, ret, next_states, next_action, next_reward, terminal, indices = \
                        frb.sample_transition_batch(batch_size=1, indices=[i])
                        states = states.transpose((0, 3, 1, 2))[0]  # (1, 84, 84, 4) --> (4, 84, 84)
                        obss.append(states)
                        actions.append(ac[0])
                        stepwise_returns.append(ret[0])
                        if terminal[0]:
                            done_idxs.append(len(obss))
                            returns.append(0)
                            if trajectories_to_load == 0:
                                done = True
                            else:
                                trajectories_to_load -= 1
                        returns[-1] += ret[0]
                        i += 1
                        if i >= 100000:
                            obss = obss[:curr_num_transitions]
                            actions = actions[:curr_num_transitions]
                            stepwise_returns = stepwise_returns[:curr_num_transitions]
                            returns[-1] = 0
                            i = transitions_per_buffer[buffer_num]
                            done = True
                    num_trajectories += (cfg.dataset.trajectories_per_buffer - trajectories_to_load)
                    transitions_per_buffer[buffer_num] = i

            actions = np.array(actions)
            returns = np.array(returns)
            stepwise_returns = np.array(stepwise_returns)
            done_idxs = np.array(done_idxs)

            # -- create reward-to-go dataset
            start_index = 0
            rtg = np.zeros_like(stepwise_returns)
            for i in done_idxs:
                i = int(i)
                curr_traj_returns = stepwise_returns[start_index:i]
                for j in range(i - 1, start_index - 1, -1):  # start from i-1
                    rtg_j = curr_traj_returns[j - start_index:i - start_index]
                    rtg[j] = sum(rtg_j)
                start_index = i

            # -- create timestep dataset
            start_index = 0
            timesteps = np.zeros(len(actions) + 1, dtype=int)
            for i in done_idxs:
                i = int(i)
                timesteps[start_index:i + 1] = np.arange(i + 1 - start_index)
                start_index = i + 1

            self.obss = obss
            self.actions = actions
            self.done_idxs = done_idxs
            self.rtgs = rtg
            self.timesteps = timesteps
            # return obss, actions, returns, done_idxs, rtg, timesteps

    def get_max_timestep(self) -> int:
        """
        Overview:
            Get the max timestep of the dataset.
        """

        return max(self.timesteps)

    def get_state_stats(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Overview:
            Get the state mean and std of the dataset.
        """

        return deepcopy(self.state_mean), deepcopy(self.state_std)

    def get_d4rl_dataset_stats(self, env_d4rl_name: str) -> Dict[str, list]:
        """
        Overview:
            Get the d4rl dataset stats.
        Arguments:
            - env_d4rl_name (:obj:`str`): The d4rl env name.
        """

        return self.D4RL_DATASET_STATS[env_d4rl_name]

    def __len__(self) -> int:
        """
        Overview:
            Get the length of the dataset.
        """

        if self.env_type != 'atari':
            return len(self.trajectories)
        else:
            return len(self.obss) - self.context_len

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Overview:
            Get the item of the dataset.
        Arguments:
            - idx (:obj:`int`): The index of the dataset.
        """

        if self.env_type != 'atari':
            traj = self.trajectories[idx]
            traj_len = traj['observations'].shape[0]

            if traj_len > self.context_len:
                # sample random index to slice trajectory
                si = np.random.randint(0, traj_len - self.context_len)

                states = torch.from_numpy(traj['observations'][si:si + self.context_len])
                actions = torch.from_numpy(traj['actions'][si:si + self.context_len])
                returns_to_go = torch.from_numpy(traj['returns_to_go'][si:si + self.context_len])
                timesteps = torch.arange(start=si, end=si + self.context_len, step=1)

                # all ones since no padding
                traj_mask = torch.ones(self.context_len, dtype=torch.long)

            else:
                padding_len = self.context_len - traj_len

                # padding with zeros
                states = torch.from_numpy(traj['observations'])
                states = torch.cat(
                    [states, torch.zeros(([padding_len] + list(states.shape[1:])), dtype=states.dtype)], dim=0
                )

                actions = torch.from_numpy(traj['actions'])
                actions = torch.cat(
                    [actions, torch.zeros(([padding_len] + list(actions.shape[1:])), dtype=actions.dtype)], dim=0
                )

                returns_to_go = torch.from_numpy(traj['returns_to_go'])
                returns_to_go = torch.cat(
                    [
                        returns_to_go,
                        torch.zeros(([padding_len] + list(returns_to_go.shape[1:])), dtype=returns_to_go.dtype)
                    ],
                    dim=0
                )

                timesteps = torch.arange(start=0, end=self.context_len, step=1)

                traj_mask = torch.cat(
                    [torch.ones(traj_len, dtype=torch.long),
                     torch.zeros(padding_len, dtype=torch.long)], dim=0
                )
            return timesteps, states, actions, returns_to_go, traj_mask
        else:  # mean cost less than 0.001s
            block_size = self.context_len
            done_idx = idx + block_size
            for i in self.done_idxs:
                if i > idx:  # first done_idx greater than idx
                    done_idx = min(int(i), done_idx)
                    break
            idx = done_idx - block_size
            states = torch.as_tensor(
                np.array(self.obss[idx:done_idx]), dtype=torch.float32
            ).view(block_size, -1)  # (block_size, 4*84*84)
            states = states / 255.
            actions = torch.as_tensor(self.actions[idx:done_idx], dtype=torch.long).unsqueeze(1)  # (block_size, 1)
            rtgs = torch.as_tensor(self.rtgs[idx:done_idx], dtype=torch.float32).unsqueeze(1)
            timesteps = torch.as_tensor(self.timesteps[idx:idx + 1], dtype=torch.int64).unsqueeze(1)
            traj_mask = torch.ones(self.context_len, dtype=torch.long)
            return timesteps, states, actions, rtgs, traj_mask


@DATASET_REGISTRY.register('d4rl_diffuser')
class D4RLDiffuserDataset(Dataset):
    """
    Overview:
        D4RL diffuser dataset, which is used for offline RL algorithms.
    Interfaces:
        ``__init__``, ``__len__``, ``__getitem__``
    """

    def __init__(self, dataset_path: str, context_len: int, rtg_scale: float) -> None:
        """
        Overview:
            Initialization method of D4RLDiffuserDataset.
        Arguments:
            - dataset_path (:obj:`str`): The dataset path.
            - context_len (:obj:`int`): The length of the context.
            - rtg_scale (:obj:`float`): The scale of the returns to go.
        """

        self.context_len = context_len

        # load dataset
        with open(dataset_path, 'rb') as f:
            self.trajectories = pickle.load(f)

        if isinstance(self.trajectories[0], list):
            # for our collected dataset, e.g. cartpole/lunarlander case
            trajectories_tmp = []

            original_keys = ['obs', 'next_obs', 'action', 'reward']
            keys = ['observations', 'next_observations', 'actions', 'rewards']
            for key, o_key in zip(keys, original_keys):
                trajectories_tmp = [
                    {
                        key: np.stack(
                            [
                                self.trajectories[eps_index][transition_index][o_key]
                                for transition_index in range(len(self.trajectories[eps_index]))
                            ],
                            axis=0
                        )
                    } for eps_index in range(len(self.trajectories))
                ]
            self.trajectories = trajectories_tmp

        states = []
        for traj in self.trajectories:
            traj_len = traj['observations'].shape[0]
            states.append(traj['observations'])
            # calculate returns to go and rescale them
            traj['returns_to_go'] = discount_cumsum(traj['rewards'], 1.0) / rtg_scale

        # used for input normalization
        states = np.concatenate(states, axis=0)
        self.state_mean, self.state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6

        # normalize states
        for traj in self.trajectories:
            traj['observations'] = (traj['observations'] - self.state_mean) / self.state_std


class FixedReplayBuffer(object):
    """
    Overview:
        Object composed of a list of OutofGraphReplayBuffers.
    Interfaces:
        ``__init__``, ``get_transition_elements``, ``sample_transition_batch``
    """

    def __init__(self, data_dir: str, replay_suffix: int, *args, **kwargs):
        """
        Overview:
            Initialize the FixedReplayBuffer class.
        Arguments:
            - data_dir (:obj:`str`): Log directory from which to load the replay buffer.
            - replay_suffix (:obj:`int`): If not None, then only load the replay buffer \
                corresponding to the specific suffix in data directory.
            - args (:obj:`list`): Arbitrary extra arguments.
            - kwargs (:obj:`dict`): Arbitrary keyword arguments.
        """

        self._args = args
        self._kwargs = kwargs
        self._data_dir = data_dir
        self._loaded_buffers = False
        self.add_count = np.array(0)
        self._replay_suffix = replay_suffix
        if not self._loaded_buffers:
            if replay_suffix is not None:
                assert replay_suffix >= 0, 'Please pass a non-negative replay suffix'
                self.load_single_buffer(replay_suffix)
            else:
                pass
            # self._load_replay_buffers(num_buffers=50)

    def load_single_buffer(self, suffix):
        """
        Overview:
            Load a single replay buffer.
        Arguments:
            - suffix (:obj:`int`): The suffix of the replay buffer.
        """

        replay_buffer = self._load_buffer(suffix)
        if replay_buffer is not None:
            self._replay_buffers = [replay_buffer]
            self.add_count = replay_buffer.add_count
            self._num_replay_buffers = 1
            self._loaded_buffers = True

    def _load_buffer(self, suffix):
        """
        Overview:
            Loads a OutOfGraphReplayBuffer replay buffer.
        Arguments:
            - suffix (:obj:`int`): The suffix of the replay buffer.
        """

        try:
            from dopamine.replay_memory import circular_replay_buffer
            STORE_FILENAME_PREFIX = circular_replay_buffer.STORE_FILENAME_PREFIX
            # pytype: disable=attribute-error
            replay_buffer = circular_replay_buffer.OutOfGraphReplayBuffer(*self._args, **self._kwargs)
            replay_buffer.load(self._data_dir, suffix)
            # pytype: enable=attribute-error
            return replay_buffer
        # except tf.errors.NotFoundError:
        except:
            raise ('can not load')

    def get_transition_elements(self):
        """
        Overview:
            Returns the transition elements.
        """

        return self._replay_buffers[0].get_transition_elements()

    def sample_transition_batch(self, batch_size=None, indices=None):
        """
        Overview:
            Returns a batch of transitions (including any extra contents).
        Arguments:
            - batch_size (:obj:`int`): The batch size.
            - indices (:obj:`list`): The indices of the batch.
        """

        buffer_index = np.random.randint(self._num_replay_buffers)
        return self._replay_buffers[buffer_index].sample_transition_batch(batch_size=batch_size, indices=indices)


class PCDataset(Dataset):
    """
    Overview:
        Dataset for Procedure Cloning.
    Interfaces:
        ``__init__``, ``__len__``, ``__getitem__``
    """

    def __init__(self, all_data):
        """
        Overview:
            Initialization method of PCDataset.
        Arguments:
            - all_data (:obj:`tuple`): The tuple of all data.
        """

        self._data = all_data

    def __getitem__(self, item):
        """
        Overview:
            Get the item of the dataset.
        Arguments:
            - item (:obj:`int`): The index of the dataset.
        """

        return {'obs': self._data[0][item], 'bfs_in': self._data[1][item], 'bfs_out': self._data[2][item]}

    def __len__(self):
        """
        Overview:
            Get the length of the dataset.
        """

        return self._data[0].shape[0]


def load_bfs_datasets(train_seeds=1, test_seeds=5):
    """
    Overview:
        Load BFS datasets.
    Arguments:
        - train_seeds (:obj:`int`): The number of train seeds.
        - test_seeds (:obj:`int`): The number of test seeds.
    """

    from dizoo.maze.envs import Maze

    def load_env(seed):
        ccc = easydict.EasyDict({'size': 16})
        e = Maze(ccc)
        e.seed(seed)
        e.reset()
        return e

    envs = [load_env(i) for i in range(train_seeds + test_seeds)]

    observations_train = []
    observations_test = []
    bfs_input_maps_train = []
    bfs_input_maps_test = []
    bfs_output_maps_train = []
    bfs_output_maps_test = []
    for idx, env in enumerate(envs):
        if idx < train_seeds:
            observations = observations_train
            bfs_input_maps = bfs_input_maps_train
            bfs_output_maps = bfs_output_maps_train
        else:
            observations = observations_test
            bfs_input_maps = bfs_input_maps_test
            bfs_output_maps = bfs_output_maps_test

        start_obs = env.process_states(env._get_obs(), env.get_maze_map())
        _, track_back = get_vi_sequence(env, start_obs)
        env_observations = torch.stack([track_back[i][0] for i in range(len(track_back))], dim=0)

        for i in range(env_observations.shape[0]):
            bfs_sequence, _ = get_vi_sequence(env, env_observations[i].numpy().astype(np.int32))  # [L, W, W]
            bfs_input_map = env.n_action * np.ones([env.size, env.size], dtype=np.long)

            for j in range(bfs_sequence.shape[0]):
                bfs_input_maps.append(torch.from_numpy(bfs_input_map))
                bfs_output_maps.append(torch.from_numpy(bfs_sequence[j]))
                observations.append(env_observations[i])
                bfs_input_map = bfs_sequence[j]

    train_data = PCDataset(
        (
            torch.stack(observations_train, dim=0),
            torch.stack(bfs_input_maps_train, dim=0),
            torch.stack(bfs_output_maps_train, dim=0),
        )
    )
    test_data = PCDataset(
        (
            torch.stack(observations_test, dim=0),
            torch.stack(bfs_input_maps_test, dim=0),
            torch.stack(bfs_output_maps_test, dim=0),
        )
    )

    return train_data, test_data


@DATASET_REGISTRY.register('bco')
class BCODataset(Dataset):
    """
    Overview:
        Dataset for Behavioral Cloning from Observation.
    Interfaces:
        ``__init__``, ``__len__``, ``__getitem__``
    Properties:
        - obs (:obj:`np.ndarray`): The observation array.
        - action (:obj:`np.ndarray`): The action array.
    """

    def __init__(self, data=None):
        """
        Overview:
            Initialization method of BCODataset.
        Arguments:
            - data (:obj:`dict`): The data dict.
        """

        if data is None:
            raise ValueError('Dataset can not be empty!')
        else:
            self._data = data

    def __len__(self):
        """
        Overview:
            Get the length of the dataset.
        """

        return len(self._data['obs'])

    def __getitem__(self, idx):
        """
        Overview:
            Get the item of the dataset.
        Arguments:
            - idx (:obj:`int`): The index of the dataset.
        """

        return {k: self._data[k][idx] for k in self._data.keys()}

    @property
    def obs(self):
        """
        Overview:
            Get the observation array.
        """

        return self._data['obs']

    @property
    def action(self):
        """
        Overview:
            Get the action array.
        """

        return self._data['action']


@DATASET_REGISTRY.register('diffuser_traj')
class SequenceDataset(torch.utils.data.Dataset):
    """
    Overview:
        Dataset for diffuser.
    Interfaces:
        ``__init__``, ``__len__``, ``__getitem__``
    """

    def __init__(self, cfg):
        """
        Overview:
            Initialization method of SequenceDataset.
        Arguments:
            - cfg (:obj:`dict`): The config dict.
        """

        import gym

        env_id = cfg.env.env_id
        data_path = cfg.policy.collect.get('data_path', None)
        env = gym.make(env_id)

        dataset = env.get_dataset()

        self.returns_scale = cfg.env.returns_scale
        self.horizon = cfg.env.horizon
        self.max_path_length = cfg.env.max_path_length
        self.discount = cfg.policy.learn.discount_factor
        self.discounts = self.discount ** np.arange(self.max_path_length)[:, None]
        self.use_padding = cfg.env.use_padding
        self.include_returns = cfg.env.include_returns
        self.env_id = cfg.env.env_id
        itr = self.sequence_dataset(env, dataset)
        self.n_episodes = 0

        fields = {}
        for k in dataset.keys():
            if 'metadata' in k:
                continue
            fields[k] = []
        fields['path_lengths'] = []

        for i, episode in enumerate(itr):
            path_length = len(episode['observations'])
            assert path_length <= self.max_path_length
            fields['path_lengths'].append(path_length)
            for key, val in episode.items():
                if key not in fields:
                    fields[key] = []
                if val.ndim < 2:
                    val = np.expand_dims(val, axis=-1)
                shape = (self.max_path_length, val.shape[-1])
                arr = np.zeros(shape, dtype=np.float32)
                arr[:path_length] = val
                fields[key].append(arr)
            if episode['terminals'].any() and cfg.env.termination_penalty and 'timeouts' in episode:
                assert not episode['timeouts'].any(), 'Penalized a timeout episode for early termination'
                fields['rewards'][-1][path_length - 1] += cfg.env.termination_penalty
            self.n_episodes += 1

        for k in fields.keys():
            fields[k] = np.array(fields[k])

        self.normalizer = DatasetNormalizer(fields, cfg.policy.normalizer, path_lengths=fields['path_lengths'])
        self.indices = self.make_indices(fields['path_lengths'], self.horizon)

        self.observation_dim = cfg.env.obs_dim
        self.action_dim = cfg.env.action_dim
        self.fields = fields
        self.normalize()
        self.normed = False
        if cfg.env.normed:
            self.vmin, self.vmax = self._get_bounds()
            self.normed = True

        # shapes = {key: val.shape for key, val in self.fields.items()}
        # print(f'[ datasets/mujoco ] Dataset fields: {shapes}')

    def sequence_dataset(self, env, dataset=None):
        """
        Overview:
            Sequence the dataset.
        Arguments:
            - env (:obj:`gym.Env`): The gym env.
        """

        import collections
        N = dataset['rewards'].shape[0]
        if 'maze2d' in env.spec.id:
            dataset = self.maze2d_set_terminals(env, dataset)
        data_ = collections.defaultdict(list)

        # The newer version of the dataset adds an explicit
        # timeouts field. Keep old method for backwards compatability.
        use_timeouts = 'timeouts' in dataset

        episode_step = 0
        for i in range(N):
            done_bool = bool(dataset['terminals'][i])
            if use_timeouts:
                final_timestep = dataset['timeouts'][i]
            else:
                final_timestep = (episode_step == env._max_episode_steps - 1)

            for k in dataset:
                if 'metadata' in k:
                    continue
                data_[k].append(dataset[k][i])

            if done_bool or final_timestep:
                episode_step = 0
                episode_data = {}
                for k in data_:
                    episode_data[k] = np.array(data_[k])
                if 'maze2d' in env.spec.id:
                    episode_data = self.process_maze2d_episode(episode_data)
                yield episode_data
                data_ = collections.defaultdict(list)

            episode_step += 1

    def maze2d_set_terminals(self, env, dataset):
        """
        Overview:
            Set the terminals for maze2d.
        Arguments:
            - env (:obj:`gym.Env`): The gym env.
            - dataset (:obj:`dict`): The dataset dict.
        """

        goal = env.get_target()
        threshold = 0.5

        xy = dataset['observations'][:, :2]
        distances = np.linalg.norm(xy - goal, axis=-1)
        at_goal = distances < threshold
        timeouts = np.zeros_like(dataset['timeouts'])

        # timeout at time t iff
        #      at goal at time t and
        #      not at goal at time t + 1
        timeouts[:-1] = at_goal[:-1] * ~at_goal[1:]

        timeout_steps = np.where(timeouts)[0]
        path_lengths = timeout_steps[1:] - timeout_steps[:-1]

        print(
            f'[ utils/preprocessing ] Segmented {env.spec.id} | {len(path_lengths)} paths | '
            f'min length: {path_lengths.min()} | max length: {path_lengths.max()}'
        )

        dataset['timeouts'] = timeouts
        return dataset

    def process_maze2d_episode(self, episode):
        """
        Overview:
            Process the maze2d episode, adds in `next_observations` field to episode.
        Arguments:
            - episode (:obj:`dict`): The episode dict.
        """

        assert 'next_observations' not in episode
        length = len(episode['observations'])
        next_observations = episode['observations'][1:].copy()
        for key, val in episode.items():
            episode[key] = val[:-1]
        episode['next_observations'] = next_observations
        return episode

    def normalize(self, keys=['observations', 'actions']):
        """
        Overview:
            Normalize the dataset, normalize fields that will be predicted by the diffusion model
        Arguments:
            - keys (:obj:`list`): The list of keys.
        """

        for key in keys:
            array = self.fields[key].reshape(self.n_episodes * self.max_path_length, -1)
            normed = self.normalizer.normalize(array, key)
            self.fields[f'normed_{key}'] = normed.reshape(self.n_episodes, self.max_path_length, -1)

    def make_indices(self, path_lengths, horizon):
        """
        Overview:
            Make indices for sampling from dataset. Each index maps to a datapoint.
        Arguments:
            - path_lengths (:obj:`np.ndarray`): The path length array.
            - horizon (:obj:`int`): The horizon.
        """

        indices = []
        for i, path_length in enumerate(path_lengths):
            max_start = min(path_length - 1, self.max_path_length - horizon)
            if not self.use_padding:
                max_start = min(max_start, path_length - horizon)
            for start in range(max_start):
                end = start + horizon
                indices.append((i, start, end))
        indices = np.array(indices)
        return indices

    def get_conditions(self, observations):
        """
        Overview:
            Get the conditions on current observation for planning.
        Arguments:
            - observations (:obj:`np.ndarray`): The observation array.
        """

        if 'maze2d' in self.env_id:
            return {'condition_id': [0, self.horizon - 1], 'condition_val': [observations[0], observations[-1]]}
        else:
            return {'condition_id': [0], 'condition_val': [observations[0]]}

    def __len__(self):
        """
        Overview:
            Get the length of the dataset.
        """

        return len(self.indices)

    def _get_bounds(self):
        """
        Overview:
            Get the bounds of the dataset.
        """

        print('[ datasets/sequence ] Getting value dataset bounds...', end=' ', flush=True)
        vmin = np.inf
        vmax = -np.inf
        for i in range(len(self.indices)):
            value = self.__getitem__(i)['returns'].item()
            vmin = min(value, vmin)
            vmax = max(value, vmax)
        print('✓')
        return vmin, vmax

    def normalize_value(self, value):
        """
        Overview:
            Normalize the value.
        Arguments:
            - value (:obj:`np.ndarray`): The value array.
        """

        # [0, 1]
        normed = (value - self.vmin) / (self.vmax - self.vmin)
        # [-1, 1]
        normed = normed * 2 - 1
        return normed

    def __getitem__(self, idx, eps=1e-4):
        """
        Overview:
            Get the item of the dataset.
        Arguments:
            - idx (:obj:`int`): The index of the dataset.
            - eps (:obj:`float`): The epsilon.
        """

        path_ind, start, end = self.indices[idx]

        observations = self.fields['normed_observations'][path_ind, start:end]
        actions = self.fields['normed_actions'][path_ind, start:end]
        done = self.fields['terminals'][path_ind, start:end]

        # conditions = self.get_conditions(observations)
        trajectories = np.concatenate([actions, observations], axis=-1)

        if self.include_returns:
            rewards = self.fields['rewards'][path_ind, start:]
            discounts = self.discounts[:len(rewards)]
            returns = (discounts * rewards).sum()
            if self.normed:
                returns = self.normalize_value(returns)
            returns = np.array([returns / self.returns_scale], dtype=np.float32)
            batch = {
                'trajectories': trajectories,
                'returns': returns,
                'done': done,
                'action': actions,
            }
        else:
            batch = {
                'trajectories': trajectories,
                'done': done,
                'action': actions,
            }

        batch.update(self.get_conditions(observations))
        return batch


def hdf5_save(exp_data, expert_data_path):
    """
    Overview:
        Save the data to hdf5.
    """

    try:
        import h5py
    except ImportError:
        import sys
        logging.warning("not found h5py package, please install it trough 'pip install h5py' ")
        sys.exit(1)
    dataset = dataset = h5py.File('%s_demos.hdf5' % expert_data_path.replace('.pkl', ''), 'w')
    dataset.create_dataset('obs', data=np.array([d['obs'].numpy() for d in exp_data]), compression='gzip')
    dataset.create_dataset('action', data=np.array([d['action'].numpy() for d in exp_data]), compression='gzip')
    dataset.create_dataset('reward', data=np.array([d['reward'].numpy() for d in exp_data]), compression='gzip')
    dataset.create_dataset('done', data=np.array([d['done'] for d in exp_data]), compression='gzip')
    dataset.create_dataset('next_obs', data=np.array([d['next_obs'].numpy() for d in exp_data]), compression='gzip')


def naive_save(exp_data, expert_data_path):
    """
    Overview:
        Save the data to pickle.
    """

    with open(expert_data_path, 'wb') as f:
        pickle.dump(exp_data, f)


def offline_data_save_type(exp_data, expert_data_path, data_type='naive'):
    """
    Overview:
        Save the offline data.
    """

    globals()[data_type + '_save'](exp_data, expert_data_path)


def create_dataset(cfg, **kwargs) -> Dataset:
    """
    Overview:
        Create dataset.
    """

    cfg = EasyDict(cfg)
    import_module(cfg.get('import_names', []))
    return DATASET_REGISTRY.build(cfg.policy.collect.data_type, cfg=cfg, **kwargs)
