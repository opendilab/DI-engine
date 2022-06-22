import os

import ding.rl_utils.efficientzero.ctree.cytree as tree
import easydict
import numpy as np
import torch
import yaml

from ding.rl_utils.efficientzero.mcts_ptree import MCTS


class MuzeroModel(torch.nn.Module):

    def __init__(self, action_num, hidden_size):
        super().__init__()
        self.action_num = action_num
        self.hidden_size = hidden_size

    def initial_inference(self, observation):
        encoded_state = observation
        batch_size = encoded_state.shape[0]

        value = torch.zeros(size=(batch_size, ))
        reward = torch.zeros(size=(batch_size, self.hidden_size))
        policy_logits = torch.zeros(size=(batch_size, self.action_num))

        output = easydict.EasyDict()
        output['value_prefix'] = value
        output['value'] = value
        output['reward_hidden'] = [
            np.array(encoded_state.unsqueeze(0).tolist()),
            np.array(encoded_state.unsqueeze(0).tolist())
        ]
        output['policy_logits'] = policy_logits
        output['hidden_state'] = encoded_state
        return output

    def recurrent_inference(self, encoded_state, reward_hidden, action):
        batch_size = encoded_state.shape[0]
        next_encoded_state = encoded_state
        value = torch.zeros(size=(batch_size, ))
        reward = torch.zeros(size=(batch_size, ))
        policy_logits = torch.zeros(size=(batch_size, self.action_num))

        output = easydict.EasyDict()
        output['value_prefix'] = value
        output['value'] = value
        output['reward_hidden'] = [
            np.array(encoded_state.unsqueeze(0).tolist()),
            np.array(encoded_state.unsqueeze(0).tolist())
        ]
        output['policy_logits'] = policy_logits
        output['hidden_state'] = np.array(encoded_state.tolist())
        return output


def check_mcts():
    default_config_path = os.path.join(os.path.dirname(__file__), 'mcts_config.yaml')
    with open(default_config_path, "r") as f:
        config = yaml.safe_load(f)

    config = easydict.EasyDict(config)
    mcts_cfg = config.MCTS
    batch_size = env_nums = mcts_cfg.batch_size

    model = MuzeroModel(action_num=mcts_cfg.action_space_size, hidden_size=mcts_cfg.hidden_size)
    stack_obs = torch.zeros(
        size=(
            batch_size,
            mcts_cfg.hidden_size,
        ), dtype=torch.float
    )
    network_output = model.initial_inference(stack_obs.float())

    hidden_state_roots = network_output['hidden_state'].tolist()
    reward_hidden_roots = network_output['reward_hidden']
    value_prefix_pool = network_output['value_prefix'].tolist()
    policy_logits_pool = network_output['policy_logits'].tolist()

    roots = tree.Roots(env_nums, mcts_cfg.action_space_size, mcts_cfg.num_simulations)
    noises = [
        np.random.dirichlet([mcts_cfg.root_dirichlet_alpha] * mcts_cfg.action_space_size).astype(np.float32).tolist()
        for _ in range(env_nums)
    ]
    roots.prepare(mcts_cfg.root_exploration_fraction, noises, value_prefix_pool, policy_logits_pool)

    MCTS(mcts_cfg).search(roots, model, hidden_state_roots, reward_hidden_roots)
    roots_distributions = roots.get_distributions()


if __name__ == "__main__":
    import cProfile

    def profile_mcts():
        for i in range(100):
            check_mcts()

    # # 直接把分析结果打印到控制台
    # cProfile.run("profile_mcts()")
    # # 把分析结果保存到文件中
    cProfile.run("profile_mcts()", filename="result.out")
    # 增加排序方式
    # cProfile.run("profile_mcts()", filename="result.out", sort="cumulative")
