import os
import gym
import numpy as np
import copy
import torch
from tensorboardX import SummaryWriter

from ding.config import compile_config
from ding.worker import BaseLearner, BattleSampleSerialCollector, BattleInteractionSerialEvaluator, NaiveReplayBuffer
from ding.envs import SyncSubprocessEnvManager, DingEnvWrapper
from ding.policy import DQNPolicy
from ding.utils import set_pkg_seed
from ding.rl_utils import get_epsilon_greedy_fn
from dizoo.gobigger.envs import GoBiggerEnv
from dizoo.gobigger.model import GoBiggerStructedNetwork
from dizoo.gobigger.config.gobigger_selfplay_no_spatial_config import main_config
from gobigger.agents import BotAgent


class RandomPolicy:

    def __init__(self, action_type_shape: int, player_num: int):
        self.action_type_shape = action_type_shape
        self.player_num = player_num

    def forward(self, data: dict) -> dict:
        return {
            env_id: {
                'action': np.random.randint(0, self.action_type_shape, size=(self.player_num))
            }
            for env_id in data.keys()
        }

    def reset(self, data_id: list = []) -> None:
        pass


class RulePolicy:

    def __init__(self, team_id: int, player_num_per_team: int):
        self.collect_data = False  # necessary
        self.team_id = team_id
        self.player_num = player_num_per_team
        start, end = team_id * player_num_per_team, (team_id + 1) * player_num_per_team
        self.bot = {str(i): BotAgent(str(i)) for i in range(start, end)}

    def forward(self, data: dict, **kwargs) -> dict:
        ret = {}
        for env_id in data.keys():
            action = []
            for o in data[env_id]:
                raw_obs = o['collate_ignore_raw_obs']
                key = raw_obs['overlap']['clone'][0]['player']
                bot = self.bot[key]
                action.append(bot.step(raw_obs))
            ret[env_id] = {'action': np.array(action)}
        return ret

    def reset(self, data_id: list = []) -> None:
        pass


def main(cfg, seed=0, max_iterations=int(1e10)):
    cfg = compile_config(
        cfg,
        SyncSubprocessEnvManager,
        DQNPolicy,
        BaseLearner,
        BattleSampleSerialCollector,
        BattleInteractionSerialEvaluator,
        NaiveReplayBuffer,
        save_cfg=True
    )
    collector_env_num, evaluator_env_num = cfg.env.collector_env_num, cfg.env.evaluator_env_num
    collector_env = SyncSubprocessEnvManager(
        env_fn=[lambda: GoBiggerEnv(cfg.env) for _ in range(collector_env_num)], cfg=cfg.env.manager
    )
    random_evaluator_env = SyncSubprocessEnvManager(
        env_fn=[lambda: GoBiggerEnv(cfg.env) for _ in range(evaluator_env_num)], cfg=cfg.env.manager
    )
    rule_evaluator_env = SyncSubprocessEnvManager(
        env_fn=[lambda: GoBiggerEnv(cfg.env) for _ in range(evaluator_env_num)], cfg=cfg.env.manager
    )

    collector_env.seed(seed)
    random_evaluator_env.seed(seed, dynamic_seed=False)
    rule_evaluator_env.seed(seed, dynamic_seed=False)
    set_pkg_seed(seed, use_cuda=cfg.policy.cuda)

    model = GoBiggerStructedNetwork(**cfg.policy.model)
    policy = DQNPolicy(cfg.policy, model=model)
    rule_collect_policy = RulePolicy(1, cfg.env.player_num_per_team)
    random_eval_policy = RandomPolicy(
        cfg.policy.model.action_type_shape, cfg.env.team_num * cfg.env.player_num_per_team
    )
    rule_eval_policy = RulePolicy(1, cfg.env.player_num_per_team)
    eps_cfg = cfg.policy.other.eps
    epsilon_greedy = get_epsilon_greedy_fn(eps_cfg.start, eps_cfg.end, eps_cfg.decay, eps_cfg.type)

    tb_logger = SummaryWriter(os.path.join('./{}/log/'.format(cfg.exp_name), 'serial'))
    learner = BaseLearner(
        cfg.policy.learn.learner, policy.learn_mode, tb_logger, exp_name=cfg.exp_name, instance_name='learner'
    )
    collector = BattleSampleSerialCollector(
        cfg.policy.collect.collector,
        collector_env, [policy.collect_mode, rule_collect_policy],
        tb_logger,
        exp_name=cfg.exp_name
    )
    random_evaluator = BattleInteractionSerialEvaluator(
        cfg.policy.eval.evaluator,
        random_evaluator_env, [policy.eval_mode, random_eval_policy],
        tb_logger,
        exp_name=cfg.exp_name,
        instance_name='random_evaluator'
    )
    rule_evaluator = BattleInteractionSerialEvaluator(
        cfg.policy.eval.evaluator,
        rule_evaluator_env, [policy.eval_mode, rule_eval_policy],
        tb_logger,
        exp_name=cfg.exp_name,
        instance_name='rule_evaluator'
    )
    replay_buffer = NaiveReplayBuffer(cfg.policy.other.replay_buffer, exp_name=cfg.exp_name)

    for _ in range(max_iterations):
        if random_evaluator.should_eval(learner.train_iter):
            random_stop_flag, random_reward, _ = random_evaluator.eval(
                learner.save_checkpoint, learner.train_iter, collector.envstep
            )
            rule_stop_flag, rule_reward, _ = rule_evaluator.eval(
                learner.save_checkpoint, learner.train_iter, collector.envstep
            )
            if random_stop_flag and rule_stop_flag:
                break
        eps = epsilon_greedy(collector.envstep)
        # Sampling data from environments
        new_data, _ = collector.collect(train_iter=learner.train_iter, policy_kwargs={'eps': eps})
        replay_buffer.push(new_data[0], cur_collector_envstep=collector.envstep)
        replay_buffer.push(new_data[1], cur_collector_envstep=collector.envstep)
        for i in range(cfg.policy.learn.update_per_collect):
            train_data = replay_buffer.sample(learner.policy.get_attribute('batch_size'), learner.train_iter)
            learner.train(train_data, collector.envstep)


if __name__ == "__main__":
    main(main_config)
