import torch
from easydict import EasyDict
from ding.model.template.efficientzero.efficientzero_base_model import inverse_scalar_transform


class MuZeroModelFake(torch.nn.Module):

    def __init__(self, action_num):
        super().__init__()
        self.action_num = action_num

    def initial_inference(self, observation):
        encoded_state = observation
        batch_size = encoded_state.shape[0]

        value = torch.zeros(size=(batch_size, 601))
        value_prefix = [0. for _ in range(batch_size)]
        policy_logits = torch.zeros(size=(batch_size, self.action_num))
        hidden_state = torch.zeros(size=(batch_size, 12, 3, 3))
        reward_hidden_state_state = (torch.zeros(size=(1, batch_size, 16)), torch.zeros(size=(1, batch_size, 16)))

        output = {
            'value': value,
            'value_prefix': value_prefix,
            'policy_logits': policy_logits,
            'hidden_state': hidden_state,
            'reward_hidden_state': reward_hidden_state_state
        }

        return EasyDict(output)

    def recurrent_inference(self, hidden_states, reward_hidden_states, actions):
        batch_size = hidden_states.shape[0]
        hidden_state = torch.zeros(size=(batch_size, 12, 3, 3))
        reward_hidden_state_state = (torch.zeros(size=(1, batch_size, 16)), torch.zeros(size=(1, batch_size, 16)))
        value = torch.zeros(size=(batch_size, 601))
        value_prefix = torch.zeros(size=(batch_size, 601))
        policy_logits = torch.zeros(size=(batch_size, self.action_num))

        output = {
            'value': value,
            'value_prefix': value_prefix,
            'policy_logits': policy_logits,
            'hidden_state': hidden_state,
            'reward_hidden_state': reward_hidden_state_state
        }

        return EasyDict(output)


def check_mcts():
    import ding.rl_utils.mcts.ptree as tree
    import numpy as np
    from ding.rl_utils.mcts.mcts_ptree import EfficientZeroMCTSPtree as MCTS

    game_config = EasyDict(
        dict(
            lstm_horizon_len=5,
            support_size=300,
            action_space_size=100,
            num_simulations=100,
            batch_size=10,
            pb_c_base=1,
            pb_c_init=1,
            discount=0.9,
            root_dirichlet_alpha=0.3,
            root_exploration_fraction=0.2,
            dirichlet_alpha=0.3,
            exploration_fraction=1,
            device='cpu',
        )
    )

    batch_size = env_nums = game_config.batch_size
    action_space_size = game_config.action_space_size

    model = MuZeroModelFake(action_num=100)
    stack_obs = torch.zeros(
        size=(
            batch_size,
            100,
        ), dtype=torch.float
    )

    network_output = model.initial_inference(stack_obs.float())

    hidden_state_roots = network_output['hidden_state']
    reward_hidden_state_state = network_output['reward_hidden_state']
    pred_values_pool = network_output['value']
    value_prefix_pool = network_output['value_prefix']
    policy_logits_pool = network_output['policy_logits']

    # network output process
    pred_values_pool = inverse_scalar_transform(pred_values_pool, game_config.support_size).detach().cpu().numpy()
    hidden_state_roots = hidden_state_roots.detach().cpu().numpy()
    reward_hidden_state_state = (
        reward_hidden_state_state[0].detach().cpu().numpy(), reward_hidden_state_state[1].detach().cpu().numpy()
    )
    policy_logits_pool = policy_logits_pool.detach().cpu().numpy().tolist()

    roots = tree.Roots(env_nums, game_config.action_space_size, game_config.num_simulations)
    noises = [
        np.random.dirichlet([game_config.root_dirichlet_alpha] * game_config.action_space_size
                            ).astype(np.float32).tolist() for _ in range(env_nums)
    ]
    roots.prepare(game_config.root_exploration_fraction, noises, value_prefix_pool, policy_logits_pool)
    MCTS(game_config).search(roots, model, hidden_state_roots, reward_hidden_state_state)
    roots_distributions = roots.get_distributions()
    assert np.array(roots_distributions).shape == (batch_size, action_space_size)


if __name__ == '__main__':
    import cProfile

    run_num = 10

    def profile_mcts(run_num):
        for i in range(run_num):
            check_mcts()

    # https://www.jianshu.com/p/fd3dbc067003

    # 直接把分析结果打印到控制台
    # cProfile.run(f"profile_mcts({run_num})")
    # 把分析结果保存到文件中
    cProfile.run(f"profile_mcts({run_num})", filename="result.out")
    # 把分析结果保存到文件中, 增加排序方式
    # cProfile.run(f"profile_mcts({run_num})", filename="result.out", sort="cumulative")
    """
    cProfile 结果分析
    """

    # 生成可视化分析图片
    # python gprof2dot.py - f pstats result.out | dot - Tpng - o result.png
    """
    import pstats
    # 创建Stats对象
    p = pstats.Stats("result.out")

    # strip_dirs(): 去掉无关的路径信息
    # sort_stats(): 排序，支持的方式和上述的一致
    # print_stats(): 打印分析结果，可以指定打印前几行

    # 和直接运行cProfile.run("test()")的结果是一样的
    # p.strip_dirs().sort_stats(-1).print_stats()

    # 按照函数名排序，只打印前3行函数的信息, 参数还可为小数,表示前百分之几的函数信息
    # p.strip_dirs().sort_stats("name").print_stats(3)

    # 按照运行时间和函数名进行排序
    p.strip_dirs().sort_stats("cumulative", "name").print_stats(0.5)

    # 如果想知道有哪些函数调用了search
    # p.print_callers(0.5, "search")

    # 查看search()函数中调用了哪些函数
    # p.print_callees("search")
    """
