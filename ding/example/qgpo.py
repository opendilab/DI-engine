import torch
import gym
import d4rl
from easydict import EasyDict
from ditk import logging
from ding.model import QGPO
from ding.policy import QGPOPolicy
from ding.envs import DingEnvWrapper, BaseEnvManagerV2
from ding.config import compile_config
from ding.framework import task, ding_init
from ding.framework.context import OfflineRLContext
from ding.framework.middleware import trainer, CkptSaver, offline_logger, wandb_offline_logger, termination_checker
from ding.framework.middleware.functional.evaluator import interaction_evaluator
from ding.framework.middleware.functional.data_processor import qgpo_support_data_generator, qgpo_offline_data_fetcher
from ding.utils import set_pkg_seed

from dizoo.d4rl.config.halfcheetah_medium_expert_qgpo_config import main_config, create_config


class QGPOD4RLDataset(torch.utils.data.Dataset):
    """
    Overview:
        Dataset for QGPO algorithm. The training of QGPO algorithm is based on contrastive energy prediction, \
        which needs true action and fake action. The true action is sampled from the dataset, and the fake action \
        is sampled from the action support generated by the behavior policy.
    Interface:
        ``__init__``, ``__getitem__``, ``__len__``.
    """

    def __init__(self, cfg, device="cpu"):
        """
        Overview:
            Initialization method of QGPOD4RLDataset class
        Arguments:
            - cfg (:obj:`EasyDict`): Config dict
            - device (:obj:`str`): Device name
        """

        self.cfg = cfg
        data = d4rl.qlearning_dataset(gym.make(cfg.env_id))
        self.device = device
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
            min_ret, max_ret = QGPOD4RLDataset.return_range(data, 1000)
            reward /= (max_ret - min_ret)
            reward *= 1000
        elif reward_tune == 'cql_antmaze':
            reward = (reward - 0.5) * 4.0
        elif reward_tune == 'antmaze':
            reward = (reward - 0.25) * 2.0
        self.rewards = reward
        self.len = self.states.shape[0]
        logging.info(f"{self.len} data loaded in QGPOD4RLDataset")

    def __getitem__(self, index):
        """
        Overview:
            Get data by index
        Arguments:
            - index (:obj:`int`): Index of data
        Returns:
            - data (:obj:`dict`): Data dict

        .. note::
            The data dict contains the following keys:
            - s (:obj:`torch.Tensor`): State
            - a (:obj:`torch.Tensor`): Action
            - r (:obj:`torch.Tensor`): Reward
            - s_ (:obj:`torch.Tensor`): Next state
            - d (:obj:`torch.Tensor`): Is finished
            - fake_a (:obj:`torch.Tensor`): Fake action for contrastive energy prediction and qgpo training \
                (fake action is sampled from the action support generated by the behavior policy)
            - fake_a_ (:obj:`torch.Tensor`): Fake next action for contrastive energy prediction and qgpo training \
                (fake action is sampled from the action support generated by the behavior policy)
        """

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

    def __len__(self):
        return self.len

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


def main():
    # If you don't have offline data, you need to prepare if first and set the data_path in config
    # For demostration, we also can train a RL policy (e.g. SAC) and collect some data
    logging.getLogger().setLevel(logging.INFO)
    cfg = compile_config(main_config, policy=QGPOPolicy)
    ding_init(cfg)
    with task.start(async_mode=False, ctx=OfflineRLContext()):
        set_pkg_seed(cfg.seed, use_cuda=cfg.policy.cuda)
        model = QGPO(cfg=cfg.policy.model)
        policy = QGPOPolicy(cfg.policy, model=model)
        dataset = QGPOD4RLDataset(cfg=cfg.dataset, device=policy._device)
        if hasattr(cfg.policy, "load_path") and cfg.policy.load_path is not None:
            policy_state_dict = torch.load(cfg.policy.load_path, map_location=torch.device("cpu"))
            policy.learn_mode.load_state_dict(policy_state_dict)

        task.use(qgpo_support_data_generator(cfg, dataset, policy))
        task.use(qgpo_offline_data_fetcher(cfg, dataset, collate_fn=None))
        task.use(trainer(cfg, policy.learn_mode))
        for guidance_scale in cfg.policy.eval.guidance_scale:
            evaluator_env = BaseEnvManagerV2(
                env_fn=[
                    lambda: DingEnvWrapper(env=gym.make(cfg.env.env_id), cfg=cfg.env, caller="evaluator")
                    for _ in range(cfg.env.evaluator_env_num)
                ],
                cfg=cfg.env.manager
            )
            task.use(interaction_evaluator(cfg, policy.eval_mode, evaluator_env, guidance_scale=guidance_scale))
        task.use(
            wandb_offline_logger(
                cfg=EasyDict(
                    dict(
                        gradient_logger=False,
                        plot_logger=True,
                        video_logger=False,
                        action_logger=False,
                        return_logger=False,
                        vis_dataset=False,
                    )
                ),
                exp_config=cfg,
                metric_list=policy._monitor_vars_learn(),
                project_name=cfg.exp_name
            )
        )
        task.use(CkptSaver(policy, cfg.exp_name, train_freq=100000))
        task.use(offline_logger())
        task.use(termination_checker(max_train_iter=500000 + cfg.policy.learn.q_value_stop_training_iter))
        task.run()


if __name__ == "__main__":
    main()
