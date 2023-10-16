from typing import Callable
import torch
import torch.nn as nn
import gym
import d4rl
import numpy as np
import sklearn
import sklearn.datasets
from sklearn.utils import shuffle as util_shuffle
import tqdm

from ditk import logging
from ding.model import QGPO
from ding.policy import QGPOPolicy
from ding.envs import DingEnvWrapper, BaseEnvManagerV2
from ding.data import create_dataset
from ding.config import compile_config
from ding.framework import task, ding_init
from ding.framework.context import OfflineRLContext
from ding.framework.middleware import interaction_evaluator, trainer, CkptSaver, offline_data_fetcher, offline_logger
from ding.utils import set_pkg_seed


# Dataset iterator
def inf_train_gen(data, batch_size=200):
    print(data)
    if data == "swissroll":
        print(data)
        data = sklearn.datasets.make_swiss_roll(n_samples=batch_size, noise=1.0)[0]
        data = data.astype("float32")[:, [0, 2]]
        data /= 5
        return data, np.sum(data ** 2, axis=-1, keepdims=True) / 9.0
    elif data == "circles":
        data = sklearn.datasets.make_circles(n_samples=batch_size, factor=.5, noise=0.08)[0]
        data = data.astype("float32")
        data *= 3
        return data
    elif data == "rings":
        n_samples4 = n_samples3 = n_samples2 = batch_size // 4
        n_samples1 = batch_size - n_samples4 - n_samples3 - n_samples2

        # so as not to have the first point = last point, we set endpoint=False
        linspace4 = np.linspace(0, 2 * np.pi, n_samples4, endpoint=False)
        linspace3 = np.linspace(0, 2 * np.pi, n_samples3, endpoint=False)
        linspace2 = np.linspace(0, 2 * np.pi, n_samples2, endpoint=False)
        linspace1 = np.linspace(0, 2 * np.pi, n_samples1, endpoint=False)

        circ4_x = np.cos(linspace4)
        circ4_y = np.sin(linspace4)
        circ3_x = np.cos(linspace4) * 0.75
        circ3_y = np.sin(linspace3) * 0.75
        circ2_x = np.cos(linspace2) * 0.5
        circ2_y = np.sin(linspace2) * 0.5
        circ1_x = np.cos(linspace1) * 0.25
        circ1_y = np.sin(linspace1) * 0.25

        X = np.vstack(
            [np.hstack([circ4_x, circ3_x, circ2_x, circ1_x]),
             np.hstack([circ4_y, circ3_y, circ2_y, circ1_y])]
        ).T * 3.0
        X = util_shuffle(X)

        center_dist = X[:, 0] ** 2 + X[:, 1] ** 2
        energy = np.zeros_like(center_dist)

        energy[(center_dist >= 8.5)] = 0.667
        energy[(center_dist >= 5.0) & (center_dist < 8.5)] = 0.333
        energy[(center_dist >= 2.0) & (center_dist < 5.0)] = 1.0
        energy[(center_dist < 2.0)] = 0.0

        # Add noise
        X = X + np.random.normal(scale=0.08, size=X.shape)

        return X.astype("float32"), energy[:, None]

    elif data == "moons":
        data, y = sklearn.datasets.make_moons(n_samples=batch_size, noise=0.1)
        data = data.astype("float32")
        data = data * 2 + np.array([-1, -0.2])
        return data.astype(np.float32), (y > 0.5).astype(np.float32)[:, None]

    elif data == "8gaussians":
        scale = 4.
        centers = [
            (0, 1),
            (-1. / np.sqrt(2), 1. / np.sqrt(2)),
            (-1, 0),
            (-1. / np.sqrt(2), -1. / np.sqrt(2)),
            (0, -1),
            (1. / np.sqrt(2), -1. / np.sqrt(2)),
            (1, 0),
            (1. / np.sqrt(2), 1. / np.sqrt(2)),
        ]

        centers = [(scale * x, scale * y) for x, y in centers]

        dataset = []
        indexes = []
        for i in range(batch_size):
            point = np.random.randn(2) * 0.5
            idx = np.random.randint(8)
            center = centers[idx]
            point[0] += center[0]
            point[1] += center[1]
            indexes.append(idx)
            dataset.append(point)
        dataset = np.array(dataset, dtype="float32")
        dataset /= 1.414
        return dataset, np.array(indexes, dtype="float32")[:, None] / 7.0

    elif data == "pinwheel":
        radial_std = 0.3
        tangential_std = 0.1
        num_classes = 5
        num_per_class = batch_size // 5
        rate = 0.25
        rads = np.linspace(0, 2 * np.pi, num_classes, endpoint=False)

        features = np.random.randn(num_classes*num_per_class, 2) \
            * np.array([radial_std, tangential_std])
        features[:, 0] += 1.
        labels = np.repeat(np.arange(num_classes), num_per_class)

        angles = rads[labels] + rate * np.exp(features[:, 0])
        rotations = np.stack([np.cos(angles), -np.sin(angles), np.sin(angles), np.cos(angles)])
        rotations = np.reshape(rotations.T, (-1, 2, 2))

        return 2 * np.random.permutation(np.einsum("ti,tij->tj", features, rotations))

    elif data == "2spirals":
        n = np.sqrt(np.random.rand(batch_size // 2, 1)) * 540 * (2 * np.pi) / 360
        d1x = -np.cos(n) * n + np.random.rand(batch_size // 2, 1) * 0.5
        d1y = np.sin(n) * n + np.random.rand(batch_size // 2, 1) * 0.5
        x = np.vstack((np.hstack((d1x, d1y)), np.hstack((-d1x, -d1y)))) / 3
        x += np.random.randn(*x.shape) * 0.1
        return x, np.clip((1 - np.concatenate([n, n]) / 10), 0, 1)

    elif data == "checkerboard":
        x1 = np.random.rand(batch_size) * 4 - 2
        x2_ = np.random.rand(batch_size) - np.random.randint(0, 2, batch_size) * 2
        x2 = x2_ + (np.floor(x1) % 2)
        points = np.concatenate([x1[:, None], x2[:, None]], 1) * 2

        points_x = points[:, 0]
        judger = ((points_x > 0) & (points_x <= 2)) | ((points_x <= -2))
        return points, judger.astype(np.float32)[:, None]

    elif data == "line":
        x = np.random.rand(batch_size) * 5 - 2.5
        y = x
        return np.stack((x, y), 1)
    elif data == "cos":
        x = np.random.rand(batch_size) * 5 - 2.5
        y = np.sin(x) * 2.5
        return np.stack((x, y), 1)
    else:
        assert False


class Toy_dataset(torch.utils.data.Dataset):

    def __init__(self, name, datanum=1000000):
        assert name in ["swissroll", "8gaussians", "moons", "rings", "checkerboard", "2spirals"]
        self.datanum = datanum
        self.name = name
        self.datas, self.energy = inf_train_gen(name, batch_size=datanum)
        self.datas = torch.Tensor(self.datas).to("cuda")
        self.energy = torch.Tensor(self.energy).to("cuda")
        self.datadim = 2

    def __getitem__(self, index):
        return {"a": self.datas[index], "e": self.energy[index]}

    def __add__(self, other):
        raise NotImplementedError

    def __len__(self):
        return self.datanum


def return_range(dataset, max_episode_steps):
    returns, lengths = [], []
    ep_ret, ep_len = 0., 0
    for r, d in zip(dataset['rewards'], dataset['terminals']):
        ep_ret += float(r)
        ep_len += 1
        if d or ep_len == max_episode_steps:
            returns.append(ep_ret)
            lengths.append(ep_len)
            ep_ret, ep_len = 0., 0
    # returns.append(ep_ret)    # incomplete trajectory
    lengths.append(ep_len)  # but still keep track of number of steps
    assert sum(lengths) == len(dataset['rewards'])
    return min(returns), max(returns)


class D4RLDataset(torch.utils.data.Dataset):

    def __init__(self, cfg):
        self.cfg = cfg
        data = d4rl.qlearning_dataset(gym.make(cfg.env_id))
        self.device = cfg.device
        self.states = torch.from_numpy(data['observations']).float().to(self.device)
        self.actions = torch.from_numpy(data['actions']).float().to(self.device)
        self.next_states = torch.from_numpy(data['next_observations']).float().to(self.device)
        reward = torch.from_numpy(data['rewards']).view(-1, 1).float().to(self.device)
        self.is_finished = torch.from_numpy(data['terminals']).view(-1, 1).float().to(self.device)

        reward_tune = "iql_antmaze" if "antmaze" in cfg.env_id else "iql_locomotion"
        if reward_tune == 'normalize':
            reward = (reward - reward.mean()) / reward.std()
        elif reward_tune == 'iql_antmaze':
            reward = reward - 1.0
        elif reward_tune == 'iql_locomotion':
            min_ret, max_ret = return_range(data, 1000)
            reward /= (max_ret - min_ret)
            reward *= 1000
        elif reward_tune == 'cql_antmaze':
            reward = (reward - 0.5) * 4.0
        elif reward_tune == 'antmaze':
            reward = (reward - 0.25) * 2.0
        self.rewards = reward
        print("dql dataloard loaded")

        self.len = self.states.shape[0]
        print(self.len, "data loaded")

    def __getitem__(self, index):
        data = {
            's': self.states[index % self.len],
            'a': self.actions[index % self.len],
            'r': self.rewards[index % self.len],
            's_': self.next_states[index % self.len],
            'd': self.is_finished[index % self.len],
            'fake_a': self.fake_actions[index % self.len]
            if hasattr(self, "fake_actions") else 0.0,  # self.fake_actions <D, 16, A>
            'fake_a_': self.fake_next_actions[index % self.len]
            if hasattr(self, "fake_next_actions") else 0.0,  # self.fake_next_actions <D, 16, A>
        }
        return data

    def __add__(self, other):
        pass

    def __len__(self):
        return self.len


main_config = dict(
    exp_name='halfcheetah_medium_expert_v2_QGPO_seed0',
    seed=0,
    env=dict(
        env_id="halfcheetah-medium-expert-v2",
        evaluator_env_num=8,
        n_evaluator_episode=8,
    ),
    dataset=dict(
        device='cuda',
        env_id="halfcheetah-medium-expert-v2",
    ),
    policy=dict(
        cuda=True,
        on_policy=False,
        load_path='./halfcheetah_medium_expert_v2_QGPO_seed0/ckpt/iteration_600000.pth.tar',
        model=dict(
            score_net=dict(
                device='cuda',
                score_base=dict(
                    device='cuda',
                    qgpo_critic=dict(
                        device='cuda',
                        alpha=3,
                        method="CEP",
                        q_alpha=1,
                    ),
                ),
            ),
            device='cuda',
            obs_dim=17,
            action_dim=6,
        ),
        learn=dict(
            learning_rate=1e-3,
            batch_size=4096,
            M=16,
            diffusion_steps=15,
            behavior_policy_stop_training_iter=0,  #600000, #1000
            energy_guided_policy_begin_training_iter=0,  #600000, #1000
        ),
        collect=dict(unroll_len=1, ),
        eval=dict(
            guidance_scale=5.0,
            diffusion_steps=15,
            evaluator=dict(eval_freq=3000, ),
        ),
    ),
)
create_config = dict()

# TASK="halfcheetah-medium-expert-v2";
# seed=0;
# setting="reproduce";
# python3
# -u train_behavior.py
# --expid $TASK${seed}${setting}
# --env $TASK
# --seed ${seed}

# TASK="halfcheetah-medium-expert-v2"; seed=0; setting="reproduce";
# python3
# -u train_critic.py
# --actor_load_path models_rl/halfcheetah-medium-expert-v20reproduce/behavior_ckpt600.pth
# --expid $TASK${seed}${setting}
# --env $TASK
# --diffusion_steps 15
# --seed ${seed}
# --alpha 3
# --q_alpha 1
# --method "CEP"


def QGPO_support_data_generator(cfg, dataset, policy) -> Callable:

    behavior_policy_stop_training_iter = cfg.policy.learn.behavior_policy_stop_training_iter if hasattr(
        cfg.policy.learn, 'behavior_policy_stop_training_iter'
    ) else np.inf
    energy_guided_policy_begin_training_iter = cfg.policy.learn.energy_guided_policy_begin_training_iter if hasattr(
        cfg.policy.learn, 'energy_guided_policy_begin_training_iter'
    ) else 0
    actions_generated = False

    def generate_fake_actions():
        policy._model.score_model.q[0].guidance_scale = 0.0
        allstates = dataset.states[:].cpu().numpy()
        actions_sampled = []
        for states in tqdm.tqdm(np.array_split(allstates, allstates.shape[0] // 4096 + 1)):
            actions_sampled.append(
                policy._model.score_model.sample(
                    states, sample_per_state=cfg.policy.learn.M, diffusion_steps=cfg.policy.learn.diffusion_steps
                )
            )
        actions = np.concatenate(actions_sampled)

        allnextstates = dataset.next_states[:].cpu().numpy()
        actions_next_states_sampled = []
        for next_states in tqdm.tqdm(np.array_split(allnextstates, allnextstates.shape[0] // 4096 + 1)):
            actions_next_states_sampled.append(
                policy._model.score_model.sample(
                    next_states, sample_per_state=cfg.policy.learn.M, diffusion_steps=cfg.policy.learn.diffusion_steps
                )
            )
        actions_next_states = np.concatenate(actions_next_states_sampled)
        return actions, actions_next_states

    def _data_generator(ctx: "OfflineRLContext"):
        nonlocal actions_generated

        if ctx.train_iter >= energy_guided_policy_begin_training_iter:
            if ctx.train_iter > behavior_policy_stop_training_iter:
                # no need to generate fake actions if fake actions are already generated
                if actions_generated:
                    pass
                else:
                    actions, actions_next_states = generate_fake_actions()
                    dataset.fake_actions = torch.Tensor(actions.astype(np.float32)).to(cfg.policy.model.device)
                    dataset.fake_next_actions = torch.Tensor(actions_next_states.astype(np.float32)
                                                             ).to(cfg.policy.model.device)
                    actions_generated = True
            else:
                # generate fake actions
                actions, actions_next_states = generate_fake_actions()
                dataset.fake_actions = torch.Tensor(actions.astype(np.float32)).to(cfg.policy.model.device)
                dataset.fake_next_actions = torch.Tensor(actions_next_states.astype(np.float32)
                                                         ).to(cfg.policy.model.device)
                actions_generated = True
        else:
            # no need to generate fake actions
            pass

    return _data_generator


def main():
    # If you don't have offline data, you need to prepare if first and set the data_path in config
    # For demostration, we also can train a RL policy (e.g. SAC) and collect some data
    logging.getLogger().setLevel(logging.INFO)
    cfg = compile_config(main_config, policy=QGPOPolicy)
    ding_init(cfg)
    with task.start(async_mode=False, ctx=OfflineRLContext()):
        set_pkg_seed(cfg.seed, use_cuda=cfg.policy.cuda)
        dataset = D4RLDataset(cfg=cfg.dataset)

        model = QGPO(cfg=cfg.policy.model)
        policy = QGPOPolicy(cfg.policy, model=model)
        if hasattr(cfg.policy, "load_path") and cfg.policy.load_path is not None:
            policy_state_dict = torch.load(cfg.policy.load_path, map_location=torch.device("cpu"))
            policy.learn_mode.load_state_dict(policy_state_dict)

        evaluator_env = BaseEnvManagerV2(
            env_fn=[
                lambda: DingEnvWrapper(env=gym.make(cfg.env.env_id), cfg=cfg.env, caller="evaluator")
                for _ in range(cfg.env.evaluator_env_num)
            ],
            cfg=cfg.env.manager
        )
        task.use(QGPO_support_data_generator(cfg, dataset, policy))
        task.use(offline_data_fetcher(cfg, dataset, collate_fn=None))
        task.use(trainer(cfg, policy.learn_mode))
        task.use(interaction_evaluator(cfg, policy.eval_mode, evaluator_env))
        task.use(CkptSaver(policy, cfg.exp_name, train_freq=100000))
        task.use(offline_logger())
        task.run()


if __name__ == "__main__":
    main()
