from pettingzoo.mpe import simple_spread_v2
import numpy as np

# 创建simple_spread_v2环境
env = simple_spread_v2.parallel_env(N=3, local_ratio=0.5, max_cycles=25, continuous_actions=False)

# 重置环境并获取初始状态
obs = env.reset()

# 打印智能体数量和动作空间
print("="*40)
print("number of agents:", env.num_agents)
print("action spaces: ", env.action_spaces)
print("observation spaces: ", env.observation_spaces)

print("action space shape of agent_0:", env.action_space('agent_0').shape)

print("="*40)
# 循环进行10个步骤
for i in range(10):
    # 针对每个智能体，随机选择一个动作并执行
    actions = {}
    for agent in env.possible_agents:
        actions.update({agent: env.action_space(agent).sample()})
    obs, rew, done, trunc, info = env.step(actions)
    print("Step:", i, "Reward:", rew)

# 关闭环境
env.close()
