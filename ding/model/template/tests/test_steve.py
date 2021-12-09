import pytest
from itertools import product
import torch
from ding.model.template.model_based.steve import EnsembleTransition, EnsembleReward, EnsembleDone
from ding.torch_utils import is_differentiable

# arguments
state_size = [16]
action_size = [16]
reward_size = [1]
done_size = [2]
ensembles = [4]

trans_args = list(product(state_size, action_size, ensembles))
rew_args = list(product(state_size, action_size, reward_size, ensembles))
done_args = list(product(state_size, action_size, done_size, ensembles))


@pytest.mark.unittest
class TestSTEVE:

    def output_check(self, model, outputs):
        if isinstance(outputs, torch.Tensor):
            loss = outputs.sum()
        elif isinstance(outputs, list):
            loss = sum([t.sum() for t in outputs])
        elif isinstance(outputs, dict):
            loss = sum([v.sum() for v in outputs.values()])
        is_differentiable(loss, model)


    @pytest.mark.parametrize('state_size, action_size, ensembles', trans_args)
    def test_EnsembleTransition(self, state_size, action_size, ensembles):
        states = torch.rand(ensembles, 10, state_size)
        actions = torch.rand(ensembles, 10, action_size)
        next_states = torch.rand(ensembles, 10, state_size)

        inputs = torch.cat([states, actions], dim=2)
        labels = next_states

        transnet = EnsembleTransition(state_size=state_size, action_size=action_size, ensemble_size=ensembles)
        mean, var = transnet(inputs)
        assert mean.size() == labels.size()


    @pytest.mark.parametrize('state_size, action_size, reward_size, ensembles', rew_args)
    def test_EnsembleReward(self, state_size, action_size, reward_size, ensembles):
        states = torch.rand(ensembles, 10, state_size)
        actions = torch.rand(ensembles, 10, action_size)
        rewards = torch.rand(ensembles, 10, reward_size)

        inputs = torch.cat([states, actions], dim=2)
        labels = rewards

        rewnet = EnsembleReward(state_size=state_size, action_size=action_size, reward_size=reward_size, ensemble_size=ensembles)
        rewards_ = rewnet(inputs)
        assert rewards_.size() == labels.size()
        self.output_check(rewnet, rewards_)


    @pytest.mark.parametrize('state_size, action_size, done_size, ensembles', done_args)
    def test_EnsembleDone(self, state_size, action_size, done_size, ensembles):
        states = torch.rand(ensembles, 10, state_size)
        actions = torch.rand(ensembles, 10, action_size)
        dones = torch.LongTensor(ensembles, 10).random_(2)

        inputs = torch.cat([states, actions], dim=2)
        labels = torch.flatten(dones, start_dim=0, end_dim=1)

        donenet = EnsembleDone(state_size=state_size, action_size=action_size, done_size=done_size, ensemble_size=ensembles)
        dones_ = donenet(inputs)
        assert dones_.size() == (ensembles, 10, done_size)
        dones_ = torch.flatten(dones_, start_dim=0, end_dim=1)

        loss = donenet.loss(dones_, labels)
        donenet.train(loss)
