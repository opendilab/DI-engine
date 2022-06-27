import torch


class MuzeroModel(torch.nn.Module):

    def __init__(self, action_num):
        super().__init__()
        self.action_num = action_num

    def initial_inference(self, observation):
        encoded_state = observation
        batch_size = encoded_state.shape[0]

        value = torch.zeros(size=(batch_size, ))
        reward = torch.zeros(size=(batch_size, ))
        policy_logits = torch.zeros(size=(batch_size, self.action_num))

        output = {}
        output['value'] = value
        output['reward'] = reward
        output['policy_logits'] = policy_logits
        output['hidden_state'] = encoded_state
        output['reward_hidden_state'] = encoded_state
        return output

    def recurrent_inference(self, hidden_state, hidden_states_reward, actions):
        batch_size = hidden_state.shape[0]
        next_encoded_state = hidden_state
        value = torch.zeros(size=(batch_size, ))
        reward = torch.zeros(size=(batch_size, ))
        policy_logits = torch.zeros(size=(batch_size, self.action_num))

        output = {}
        output['value'] = value
        output['reward'] = reward
        output['value_prefix'] = reward
        output['policy_logits'] = policy_logits
        output['hidden_state'] = next_encoded_state
        output['reward_hidden_state'] = next_encoded_state

        return output


def check_mcts():
    import os
    import yaml
    import easydict
    import ding.rl_utils.muzero.ptree as tree
    import numpy as np
    from ding.rl_utils.muzero.mcts_ptree_test import MCTS

    default_config_path = os.path.join(os.path.dirname(__file__), 'mcts_config.yaml')
    with open(default_config_path, "r") as f:
        config = yaml.safe_load(f)

    config = easydict.EasyDict(config)
    mcts_cfg = config.MCTS
    batch_size = env_nums = mcts_cfg.batch_size

    model = MuzeroModel(action_num=10)
    stack_obs = torch.zeros(
        size=(
            batch_size,
            10,
        ), dtype=torch.float
    )
    network_output = model.initial_inference(stack_obs.float())

    hidden_state_roots = network_output['hidden_state']
    reward_hidden_state_roots = network_output['reward_hidden_state']

    reward_pool = network_output['reward']
    value_pool = network_output['value']
    policy_logits_pool = network_output['policy_logits'].tolist()

    roots = tree.Roots(env_nums, mcts_cfg.action_space_size, mcts_cfg.num_simulations)
    noises = [
        np.random.dirichlet([mcts_cfg.root_dirichlet_alpha] * mcts_cfg.action_space_size).astype(np.float32).tolist()
        for _ in range(env_nums)
    ]
    roots.prepare(mcts_cfg.root_exploration_fraction, noises, reward_pool, policy_logits_pool)

    MCTS(mcts_cfg).search(roots, model, hidden_state_roots, reward_hidden_state_roots)
    roots_distributions = roots.get_distributions()
    print(roots_distributions)


if __name__ == '__main__':

    check_mcts()

    # import cProfile
    # def profile_mcts():
    #     for i in range(1):
    #         check_mcts()

    # # 直接把分析结果打印到控制台
    # cProfile.run("profile_mcts()")
    # # 把分析结果保存到文件中
    # cProfile.run("profile_mcts()", filename="result.out")
    # 增加排序方式
    # cProfile.run("profile_mcts()", filename="result.out", sort="cumulative")
