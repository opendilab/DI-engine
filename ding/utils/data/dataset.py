from typing import List, Dict, Tuple
import pickle
import torch
import numpy as np
from ditk import logging
from copy import deepcopy

from easydict import EasyDict
from torch.utils.data import Dataset

from ding.utils import DATASET_REGISTRY, import_module
from ding.rl_utils import discount_cumsum


@DATASET_REGISTRY.register('naive')
class NaiveRLDataset(Dataset):

    def __init__(self, cfg) -> None:
        assert type(cfg) in [str, EasyDict], "invalid cfg type: {}".format(type(cfg))
        if isinstance(cfg, EasyDict):
            self._data_path = cfg.policy.collect.data_path
        elif isinstance(cfg, str):
            self._data_path = cfg
        with open(self._data_path, 'rb') as f:
            self._data: List[Dict[str, torch.Tensor]] = pickle.load(f)

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self._data[idx]


@DATASET_REGISTRY.register('d4rl')
class D4RLDataset(Dataset):

    def __init__(self, cfg: dict) -> None:
        import gym
        try:
            import d4rl  # register d4rl enviroments with open ai gym
        except ImportError:
            logging.warning("not found d4rl env, please install it, refer to https://github.com/rail-berkeley/d4rl")

        # Init parameters
        data_path = cfg.policy.collect.get('data_path', None)
        env_id = cfg.env.env_id

        # Create the environment
        if data_path:
            d4rl.set_dataset_path(data_path)
        env = gym.make(env_id)
        dataset = d4rl.qlearning_dataset(env)
        if cfg.policy.collect.get('normalize_states', None):
            dataset = self._normalize_states(dataset)
        self._data = []
        self._load_d4rl(dataset)

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self._data[idx]

    def _load_d4rl(self, dataset: Dict[str, np.ndarray]) -> None:
        for i in range(len(dataset['observations'])):
            trans_data = {}
            trans_data['obs'] = torch.from_numpy(dataset['observations'][i])
            trans_data['next_obs'] = torch.from_numpy(dataset['next_observations'][i])
            trans_data['action'] = torch.from_numpy(dataset['actions'][i])
            trans_data['reward'] = torch.tensor(dataset['rewards'][i])
            trans_data['done'] = dataset['terminals'][i]
            self._data.append(trans_data)

    def _normalize_states(self, dataset, eps=1e-3):
        self._mean = dataset['observations'].mean(0, keepdims=True)
        self._std = dataset['observations'].std(0, keepdims=True) + eps
        dataset['observations'] = (dataset['observations'] - self._mean) / self._std
        dataset['next_observations'] = (dataset['next_observations'] - self._mean) / self._std
        return dataset

    @property
    def mean(self):
        return self._mean

    @property
    def std(self):
        return self._std


@DATASET_REGISTRY.register('hdf5')
class HDF5Dataset(Dataset):

    def __init__(self, cfg: dict) -> None:
        try:
            import h5py
        except ImportError:
            logging.warning("not found h5py package, please install it trough 'pip install h5py' ")
        data_path = cfg.policy.collect.get('data_path', None)
        data = h5py.File(data_path, 'r')
        self._load_data(data)
        if cfg.policy.collect.get('normalize_states', None):
            self._normalize_states()

    def __len__(self) -> int:
        return len(self._data['obs'])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {k: self._data[k][idx] for k in self._data.keys()}

    def _load_data(self, dataset: Dict[str, np.ndarray]) -> None:
        self._data = {}
        for k in dataset.keys():
            logging.info(f'Load {k} data.')
            self._data[k] = dataset[k][:]

    def _normalize_states(self, eps=1e-3):
        self._mean = self._data['obs'].mean(0, keepdims=True)
        self._std = self._data['obs'].std(0, keepdims=True) + eps
        self._data['obs'] = (self._data['obs'] - self._mean) / self._std
        self._data['next_obs'] = (self._data['next_obs'] - self._mean) / self._std

    @property
    def mean(self):
        return self._mean

    @property
    def std(self):
        return self._std


@DATASET_REGISTRY.register('d4rl_trajectory')
class D4RLTrajectoryDataset(Dataset):

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

    def __init__(self, dataset_path: str, context_len: int, rtg_scale: float) -> None:

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
                                self.trjectories[eps_index][transition_index][o_key]
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

    def get_state_stats(self) -> Tuple[np.ndarray, np.ndarray]:
        return deepcopy(self.state_mean), deepcopy(self.state_std)

    def get_d4rl_dataset_stats(self, env_d4rl_name: str) -> Dict[str, list]:
        return self.D4RL_DATASET_STATS[env_d4rl_name]

    def __len__(self) -> int:
        return len(self.trajectories)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        traj = self.trajectories[idx]
        traj_len = traj['observations'].shape[0]

        if traj_len >= self.context_len:
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


def hdf5_save(exp_data, expert_data_path):
    try:
        import h5py
    except ImportError:
        logging.warning("not found h5py package, please install it trough 'pip install h5py' ")
    import numpy as np
    dataset = dataset = h5py.File('%s_demos.hdf5' % expert_data_path.replace('.pkl', ''), 'w')
    dataset.create_dataset('obs', data=np.array([d['obs'].numpy() for d in exp_data]), compression='gzip')
    dataset.create_dataset('action', data=np.array([d['action'].numpy() for d in exp_data]), compression='gzip')
    dataset.create_dataset('reward', data=np.array([d['reward'].numpy() for d in exp_data]), compression='gzip')
    dataset.create_dataset('done', data=np.array([d['done'] for d in exp_data]), compression='gzip')
    dataset.create_dataset('next_obs', data=np.array([d['next_obs'].numpy() for d in exp_data]), compression='gzip')


def naive_save(exp_data, expert_data_path):
    with open(expert_data_path, 'wb') as f:
        pickle.dump(exp_data, f)


def offline_data_save_type(exp_data, expert_data_path, data_type='naive'):
    globals()[data_type + '_save'](exp_data, expert_data_path)


def create_dataset(cfg, **kwargs) -> Dataset:
    cfg = EasyDict(cfg)
    import_module(cfg.get('import_names', []))
    return DATASET_REGISTRY.build(cfg.policy.collect.data_type, cfg=cfg, **kwargs)
