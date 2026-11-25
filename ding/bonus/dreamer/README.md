# Dreamer V1、V2、V3 实践

本案例用 PyTorch 实现低维世界模型。在 `CartPole-v1` 和离散 `LunarLander-v3` 上，智能体从向量观测和环境奖励开始学习。采集轨迹时，RSSM 用真实观测更新后验状态。学习策略时，RSSM 只用先验滚动出未来状态，actor 和 critic 在想象轨迹上训练。

| 版本 | 世界模型 | 策略与价值学习 |
| --- | --- | --- |
| V1 | 高斯潜变量，联合 KL 在全批次平均后应用 3 nats 的下限 | λ-return 的梯度通过想象状态、奖励和折扣回到 actor。离散动作采用直通 one-hot。critic 用在线价值 bootstrap。 |
| V2 | 多因子分类潜变量和直通采样。KL 的两项均为 `KL(q ∥ p)`，更新先验和后验的权重分别为 0.8、0.2。 | 支持论文的 REINFORCE 与路径梯度混合目标，使用 critic baseline，目标 critic 每 100 次行为更新同步。`--v2-dynamics-mix` 对应路径梯度占比 `1 − ρ`。 |
| V3 | 向量观测使用 symlog。每个位置的动力学和表示 KL 分别应用 1 nat 的下限，按 Nature 版本以 1、0.1 加权。分类潜变量和策略均混合 1% 的均匀分布。 | 奖励与价值采用 symexp two-hot。优势除以至少为 1 的回报分位数 EMA 范围。在线 critic bootstrap，慢速 critic 提供正则。真实回放上的辅助价值损失也训练表示。 |

这里采用小型 MLP 和单层 GRU，V1/V2 可以用随机预采样的观测统计量做固定归一化。V1 的动作梯度沿用论文中离散实验的直通估计器。V3 保留论文中的关键损失和数值变换，GRU、层宽与优化器采用适合向量环境的 PyTorch 配置。

参考 [Dreamer V1 论文](https://arxiv.org/abs/1912.01603)、[V1 官方实现](https://github.com/google-research/dreamer)、[Dreamer V2 论文](https://arxiv.org/abs/2010.02193)、[V2 官方实现](https://github.com/danijar/dreamerv2/tree/07d906e9c4322c6fc2cd6ed23e247ccd6b7c8c41)、[Dreamer V3 Nature 论文](https://www.nature.com/articles/s41586-025-08744-2)、[V3 官方实现](https://github.com/danijar/dreamerv3/tree/e01491fad6434b2245a3b8ca201dd7faedcc458c) 和 [DI-engine PR #879](https://github.com/opendilab/DI-engine/pull/879/changes)。

## 运行

在仓库根目录使用已有 PyTorch 的 Python 环境。实验依赖 Gymnasium 1.2.3 和 Box2D。

```bash
python -m pip install 'gymnasium[box2d]==1.2.3' 'numpy==1.26.4' matplotlib
python ding/bonus/dreamer/train.py --env CartPole-v1 --versions v1 --seeds 0 \
  --workdir runs/dreamer-cartpole-example
```

完整矩阵包含两个环境、三个版本和种子 0、1、2。默认 CPU 使用单线程，每 4 次策略决策更新一次，每次采样 16 条长度 16 的序列。可通过 `--device cuda:0` 使用 GPU，或在不同进程里给每个组合分配设备。各组合写入各自的目录。

```bash
python ding/bonus/dreamer/train.py --workdir runs/dreamer-matrix
python ding/bonus/dreamer/plot_returns.py --workdir runs/dreamer-matrix
```

`train.py --help` 列出步数、采样和评估设置。CartPole 默认预算为 60,000 次策略决策，每 2,000 次评估。LunarLander 默认预算为 260,000 次，每 5,000 次评估。`--full-budget` 可保留到预算结束的完整曲线。

LunarLander 的 V1、V2 每次决策最多重复同一动作 3 个环境步，V3 最多重复 2 步。回合结束即停止，累计奖励和原始环境步数均按实际调用统计。该环境的 V2 使用 90% 路径梯度、10% REINFORCE，actor 学习率为 0.00008。CartPole 的 V2 使用 REINFORCE，KL 权重为 0.1。

## 如何判断收敛

训练、周期评估和最终确认使用各自的随机序列。周期评估每次使用相同的 20 个环境种子和贪心动作，RSSM 后验按固定的 Torch 随机序列采样。连续两个周期评估达到环境均分门槛后，在独立的 100 局上确认。CartPole 的门槛是 475，离散 LunarLander 的门槛是 200。各局得分均使用环境返回的原始奖励。确认失败时，训练继续至下一次确认或用尽预算。

每个组合目录保存 `config.json`、`metrics/eval_returns.csv`、包含确认评估逐局回报的 `result.json`、以及供推理使用的 `policy.pt`。图的横轴是原始训练环境步数，不含评估交互。聚合曲线在种子共有的步数区间内插值后计算均值和标准差。

## 复现书籍示例

以下命令使用实测达标的训练种子。CartPole 使用单线程 CPU。LunarLander 使用 CUDA，回报按 Gymnasium 1.2.3 和 Box2D 2.3.10 计算。实验使用 Python 3.11、PyTorch 2.6.0、NumPy 1.26.4 和 NVIDIA L20X。

```bash
python ding/bonus/dreamer/train.py --env CartPole-v1 --versions v1 v2 v3 --seeds 0 \
  --workdir runs/dreamer-book
python ding/bonus/dreamer/train.py --env LunarLander-v3 --versions v1 v2 --seeds 42 \
  --device cuda:0 --workdir runs/dreamer-book
python ding/bonus/dreamer/train.py --env LunarLander-v3 --versions v3 --seeds 0 \
  --device cuda:0 --workdir runs/dreamer-book
python ding/bonus/dreamer/plot_returns.py --workdir runs/dreamer-book
```

V1 在 LunarLander 的种子敏感性最明显。另一个实测达标的配置是训练种子 2 加 `--critic-lr 0.00008`，采用 V1 论文的 critic 学习率。训练日志中的 `step` 是决策次数，表中和学习曲线中的环境步数以实际 `env.step()` 调用为准。

## 实测收敛

| 环境 | 版本 | 达标种子数 | 达标种子的独立 100 局均分 | 确认时的原始环境步数 |
| --- | --- | ---: | ---: | ---: |
| CartPole-v1 | V1 | 3 / 3 | 495.64–500.00 | 10,000–30,000 |
| CartPole-v1 | V2 | 3 / 3 | 495.80–500.00 | 18,000–42,000 |
| CartPole-v1 | V3 | 3 / 3 | 500.00 | 24,000–34,000 |
| LunarLander-v3 | V1 | 1 / 7 | 223.23 | 283,968 |
| LunarLander-v3 | V2 | 3 / 4 | 205.49–232.01 | 238,682–506,625 |
| LunarLander-v3 | V3 | 3 / 3 | 213.69–231.63 | 129,789–169,663 |

CartPole 的训练种子为 0、1、2。LunarLander V1 的默认学习率在种子 42 达标，种子 0–2 的预算是 220,000 次决策，种子 3–5 的预算是 260,000 次。种子 2 使用上面的 critic 学习率变体，在 747,549 个原始环境步时取得 217.06 分。LunarLander V2 的种子 1、2、42 达标，种子 0 在 260,000 次决策内未达到确认门槛。LunarLander V3 的种子 0、1、2 均达标。

同一 CPU 核心上的微基准以 16 条长度为 16 的序列和 15 步想象视界进行，统计预热后的 150 次更新。PR #879 到当前实现的中位数，V1 为 83.6 → 85.3 ms，V2 的 REINFORCE 配置为 164.0 → 83.5 ms，V3 为 194.2 → 131.5 ms。LunarLander V2 的混合梯度路径为 137.9 ms。

## 阅读代码

- `dreamer/rssm.py` 里的观测步骤用 `a[t-1]` 推进到 `h[t]`，用 `x[t]` 推断 `z[t]`。先验和后验使用同一个 `h[t]`。
- `dreamer/replay.py` 均匀抽取单个回合内的连续序列。首个真实观测的动作占位会变成全零向量。每个训练片段的第一帧为 RSSM 提供上下文。
- `dreamer/agent.py` 的 `lambda_return` 用进入下一状态时获得的奖励和 continuation。环境达到时间上限时会重置采集状态，并保留价值 bootstrap。真正终止时折扣为零。
