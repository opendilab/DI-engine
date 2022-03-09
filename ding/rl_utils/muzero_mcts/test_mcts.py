import torch

class MuzeroModel(torch.nn.Module):
    def __init__(self, action_num):
        super().__init__()
        self.action_num = action_num

    def initial_inference(self, observation):
        encoded_state = observation
        batch_size = encoded_state.shape[0]

        value = torch.zeros(size=(batch_size,))
        reward = torch.zeros(size=(batch_size,))
        policy_logits = torch.zeros(size=(batch_size, self.action_num))

        output = {}
        output['value'] = value
        output['reward'] = reward
        output['policy_logits'] = policy_logits
        output['hidden_state'] = encoded_state
        return output
    def recurrent_inference(self, encoded_state, action):
        batch_size = encoded_state.shape[0]
        next_encoded_state = encoded_state
        value = torch.zeros(size=(batch_size,))
        reward = torch.zeros(size=(batch_size,))
        policy_logits = torch.zeros(size=(batch_size, self.action_num))

        output = {}
        output['value'] = value
        output['reward'] = reward
        output['policy_logits'] = policy_logits
        output['hidden_state'] = next_encoded_state
        return output


if __name__ == '__main__':
    import os
    import yaml
    import easydict
    import ding.rl_utils.muzero_mcts.ptree as tree
    import numpy as np
    from ding.rl_utils.muzero_mcts.mcts import MCTS

    default_config_path = os.path.join(os.path.dirname(__file__), 'mcts_config.yaml')
    with open(default_config_path, "r") as f:
        config = yaml.safe_load(f)

    config = easydict.EasyDict(config)
    mcts_cfg = config.MCTS
    batch_size = env_nums = mcts_cfg.batch_size

    model = MuzeroModel(action_num=100)
    stack_obs = torch.zeros(size=(batch_size, 100,), dtype=torch.float)
    network_output = model.initial_inference(stack_obs.float())

    hidden_state_roots = network_output['hidden_state']
    reward_pool = network_output['reward']
    value_pool = network_output['value']
    policy_logits_pool = network_output['policy_logits'].tolist()

    roots = tree.Roots(env_nums, mcts_cfg.action_space_size, mcts_cfg.num_simulations)
    noises = [np.random.dirichlet([mcts_cfg.root_dirichlet_alpha] * mcts_cfg.action_space_size).astype(
        np.float32).tolist() for _ in range(env_nums)]
    roots.prepare(mcts_cfg.root_exploration_fraction, noises, reward_pool, policy_logits_pool)

    MCTS(mcts_cfg).search(roots, model, hidden_state_roots,)
    roots_distributions = roots.get_distributions()
