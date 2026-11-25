from dataclasses import asdict

import numpy as np
import pytest
import torch
from torch.distributions import Categorical, kl_divergence

from ding.bonus.dreamer.config import DreamerConfig, TrainConfig
from ding.bonus.dreamer.dreamer.agent import discount_cumprod, lambda_return, make_agent
from ding.bonus.dreamer.dreamer.models import SymexpTwoHot, categorical_log_probs, symlog, symexp
from ding.bonus.dreamer.dreamer.replay import ReplayBuffer
from ding.bonus.dreamer.train import evaluate, step_env, train


def small_agent(version):
    config = DreamerConfig(4, 2, embed_dim=8, deter_dim=8, stoch_dim=4, stoch_classes=4, hidden_dim=8, horizon=3)
    return make_agent(version, config, torch.device("cpu"))


def test_lambda_return_uses_next_values_and_stops_at_terminal():
    r = torch.tensor([1., 2., 3.])
    v = torch.tensor([10., 20., 30.])
    d = torch.tensor([0.5, 0., 0.5])
    torch.testing.assert_close(lambda_return(r, v, d, 0.5), torch.tensor([4., 2., 18.]))
    torch.testing.assert_close(discount_cumprod(d), torch.tensor([1., 0.5, 0.]))


@pytest.mark.parametrize(
    "outputs", [
        (np.zeros(4), 1., False, True, {}),
        (np.zeros(4), 1., True, {
            "TimeLimit.truncated": True
        }),
    ]
)
def test_time_limit_ends_episode_with_continuation(outputs):

    class TimeLimitEnv:

        def step(self, action):
            return outputs

    _, _, done, terminal, _ = step_env(TimeLimitEnv(), 0)
    assert done and not terminal


def test_replay_windows_keep_actions_and_rewards_aligned():
    replay = ReplayBuffer(20, seed=4)
    replay.add_episode([[0], [1], [2], [3]], [-1, 0, 1, 0], [0, 10, 20, 30], [0, 0, 0, 1])
    replay.add_episode([[10], [11], [12]], [-1, 1, 0], [0, 110, 120], [0, 0, 0])
    batch = replay.sample(256, 3)
    starts = batch["obs"][:, 0, 0]
    assert set(starts) == {0, 1, 10}
    assert (np.diff(batch["obs"], axis=1) == 1).all()
    assert (batch["actions"][starts == 1] == [0, 1, 0]).all()
    assert (batch["dones"][starts == 1] == [0, 0, 1]).all()
    assert (batch["actions"][starts == 10] == [-1, 1, 0]).all()
    assert (batch["rewards"][starts == 10] == [0, 110, 120]).all()
    assert (batch["dones"][starts == 10] == 0).all()
    assert not replay.can_sample(5)


def test_categorical_straight_through_and_unimix():
    torch.manual_seed(3)
    model = small_agent("v3").world_model
    rssm = model.rssm
    prev = rssm.init_state(4, torch.device("cpu"))
    state = rssm.img_step(prev, torch.zeros(4, 2))
    factors = state.stoch.reshape(4, rssm.stoch_dim, rssm.stoch_classes)
    assert ((factors == 0) | (factors == 1)).all()
    torch.testing.assert_close(factors.sum(-1), torch.ones_like(factors.sum(-1)))
    factors[..., 0].sum().backward()
    assert rssm.prior_layer[-1].weight.grad.abs().sum() > 0
    log_probs = categorical_log_probs(torch.tensor([[100., -100.]]), model.config.unimix)
    torch.testing.assert_close(log_probs.exp(), torch.tensor([[0.995, 0.005]]))


def test_v2_balancing_differentiates_correct_kl_arguments():
    model = small_agent("v2").world_model
    post_logits = torch.tensor([[[0.7, -0.2, 0.1, -1.]]], requires_grad=True)
    prior_logits = torch.tensor([[[0.05, 0.6, -0.4, 0.1]]], requires_grad=True)
    loss = model.kl_loss(post_logits.log_softmax(-1), prior_logits.log_softmax(-1))
    grads = torch.autograd.grad(loss, (post_logits, prior_logits))
    expected = kl_divergence(Categorical(logits=post_logits), Categorical(logits=prior_logits)).mean()
    ref_grads = torch.autograd.grad(expected, (post_logits, prior_logits))
    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(grads[0], 0.2 * ref_grads[0])
    torch.testing.assert_close(grads[1], 0.8 * ref_grads[1])


def test_v3_applies_free_bits_per_position():
    model = small_agent("v3").world_model
    post = torch.tensor([[[0., 0.]], [[5., -5.]]]).log_softmax(-1)
    prior = torch.tensor([[[0., 0.]], [[-5., 5.]]]).log_softmax(-1)
    divergence = kl_divergence(Categorical(logits=post), Categorical(logits=prior)).squeeze(-1)
    expected = (1 + 0.1) * (1 + divergence[1]) / 2
    torch.testing.assert_close(model.kl_loss(post, prior), expected)


def test_symexp_twohot_interpolates_in_reward_units():
    head = SymexpTwoHot(255)
    mid = head.bins.numel() // 2
    low, high = head.bins[mid + 16:mid + 18]
    target = torch.stack([low + 0.25 * (high - low), head.bins[0], head.bins[-1]])
    logits = torch.randn(3, head.bins.numel(), generator=torch.Generator().manual_seed(5))
    log_probs = logits.log_softmax(-1)
    expected = -torch.stack(
        [0.75 * log_probs[0, mid + 16] + 0.25 * log_probs[0, mid + 17], log_probs[1, 0], log_probs[2, -1]]
    )
    torch.testing.assert_close(head.loss(logits, target), expected)
    torch.testing.assert_close(head.mean(torch.zeros_like(logits)), torch.zeros(3), rtol=0, atol=0)
    torch.testing.assert_close(symexp(symlog(torch.tensor([-500., 0., 250.]))), torch.tensor([-500., 0., 250.]))


@pytest.mark.parametrize("version", ["v1", "v2", "v3"])
def test_actor_gradient_is_isolated_and_update_is_finite(version, monkeypatch):
    torch.manual_seed(0)
    agent = small_agent(version)
    if version == "v3":
        # V3 初始输出层为零。模拟一次更新后，才能检查 critic 对状态特征的梯度。
        with torch.no_grad():
            agent.critic[-1].weight[agent.config.bins // 2 + 1].fill_(0.02)
    obs = torch.randn(2, 4, 4)
    actions = torch.zeros(2, 4, 2)
    actions[:, 1:, 0] = 1
    rewards = torch.ones(2, 4)
    terminals = torch.zeros(2, 4)
    _, _, posts = agent.world_model.loss(obs, actions, rewards, terminals, agent.init_state(2))

    imagine = agent._imagine

    def check_imagine(start, *, pathwise=False):
        features, acts, entropy = imagine(start, pathwise=pathwise)
        assert pathwise == (version == "v1")
        assert features.requires_grad == pathwise
        return features, acts, entropy

    monkeypatch.setattr(agent, "_imagine", check_imagine)
    actor_loss, _, aux_loss = agent._behavior_loss(posts, rewards, terminals)
    actor_loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in agent.actor.parameters())
    assert all(p.grad is None for p in agent.world_model.parameters())
    assert all(p.grad is None for p in agent.critic.parameters())
    if version == "v3":
        # replay value 梯度应该训练观测表示。
        aux_loss.backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in agent.world_model.encoder.parameters())

    batch = dict(
        obs=obs.numpy(), actions=np.zeros((2, 4), dtype=np.int64), rewards=rewards.numpy(), dones=terminals.numpy()
    )
    metrics = agent.train_step(batch)
    assert np.isfinite(list(asdict(metrics).values())).all()


def test_evaluation_keeps_training_rng_state():
    agent = small_agent("v2")
    rng_state = torch.random.get_rng_state()
    first = evaluate(agent, "CartPole-v1", 4, 123, workers=2)
    second = evaluate(agent, "CartPole-v1", 4, 123, workers=2)
    np.testing.assert_array_equal(first, second)
    torch.testing.assert_close(torch.random.get_rng_state(), rng_state)


@pytest.mark.parametrize("env_id", ["CartPole-v1", "LunarLander-v3"])
def test_training_emits_config_checkpoint_and_raw_returns(tmp_path, env_id):
    cfg = TrainConfig(
        total_steps=128,
        seed_steps=64,
        eval_every=64,
        batch_size=2,
        seq_len=4,
        pretrain_steps=1,
        eval_episodes=2,
        holdout_episodes=3,
        workdir=str(tmp_path),
        plot=False
    )
    result = train(cfg, env_id, "v3", 42)
    run = tmp_path / env_id / "v3" / "seed_42"
    assert result["updates"] > 0
    assert result["env_steps"] == 128
    assert len(result["holdouts"][-1]["returns"]) == 3
    assert (run / "config.json").exists() and (run / "result.json").exists()
    policy = torch.load(run / "policy.pt", weights_only=True)
    assert policy["env_steps"] == result["env_steps"] and policy["version"] == "v3"
