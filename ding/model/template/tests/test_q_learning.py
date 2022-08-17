import pytest
from itertools import product
import torch
from ding.model.template import DQN, RainbowDQN, QRDQN, IQN, FQF, DRQN, C51DQN
from ding.torch_utils import is_differentiable

T, B = 3, 4
obs_shape = [4, (8, ), (4, 64, 64)]
act_shape = [3, (6, ), [2, 3, 6]]
args = list(product(*[obs_shape, act_shape]))


@pytest.mark.unittest
class TestQLearning:

    def output_check(self, model, outputs):
        if isinstance(outputs, torch.Tensor):
            loss = outputs.sum()
        elif isinstance(outputs, list):
            loss = sum([t.sum() for t in outputs])
        elif isinstance(outputs, dict):
            loss = sum([v.sum() for v in outputs.values()])
        is_differentiable(loss, model)

    @pytest.mark.parametrize('obs_shape, act_shape', args)
    def test_dqn(self, obs_shape, act_shape):
        if isinstance(obs_shape, int):
            inputs = torch.randn(B, obs_shape)
        else:
            inputs = torch.randn(B, *obs_shape)
        model = DQN(obs_shape, act_shape)
        outputs = model(inputs)
        assert isinstance(outputs, dict)
        if isinstance(act_shape, int):
            assert outputs['logit'].shape == (B, act_shape)
        elif len(act_shape) == 1:
            assert outputs['logit'].shape == (B, *act_shape)
        else:
            for i, s in enumerate(act_shape):
                assert outputs['logit'][i].shape == (B, s)
        self.output_check(model, outputs['logit'])

    @pytest.mark.parametrize('obs_shape, act_shape', args)
    def test_rainbowdqn(self, obs_shape, act_shape):
        if isinstance(obs_shape, int):
            inputs = torch.randn(B, obs_shape)
        else:
            inputs = torch.randn(B, *obs_shape)
        model = RainbowDQN(obs_shape, act_shape, n_atom=41)
        outputs = model(inputs)
        assert isinstance(outputs, dict)
        if isinstance(act_shape, int):
            assert outputs['logit'].shape == (B, act_shape)
            assert outputs['distribution'].shape == (B, act_shape, 41)
        elif len(act_shape) == 1:
            assert outputs['logit'].shape == (B, *act_shape)
            assert outputs['distribution'].shape == (B, *act_shape, 41)
        else:
            for i, s in enumerate(act_shape):
                assert outputs['logit'][i].shape == (B, s)
                assert outputs['distribution'][i].shape == (B, s, 41)
        self.output_check(model, outputs['logit'])

    @pytest.mark.parametrize('obs_shape, act_shape', args)
    def test_c51(self, obs_shape, act_shape):
        if isinstance(obs_shape, int):
            inputs = torch.randn(B, obs_shape)
        else:
            inputs = torch.randn(B, *obs_shape)
        model = C51DQN(obs_shape, act_shape, n_atom=41)
        outputs = model(inputs)
        assert isinstance(outputs, dict)
        if isinstance(act_shape, int):
            assert outputs['logit'].shape == (B, act_shape)
            assert outputs['distribution'].shape == (B, act_shape, 41)
        elif len(act_shape) == 1:
            assert outputs['logit'].shape == (B, *act_shape)
            assert outputs['distribution'].shape == (B, *act_shape, 41)
        else:
            for i, s in enumerate(act_shape):
                assert outputs['logit'][i].shape == (B, s)
                assert outputs['distribution'][i].shape == (B, s, 41)
        self.output_check(model, outputs['logit'])

    @pytest.mark.parametrize('obs_shape, act_shape', args)
    def test_iqn(self, obs_shape, act_shape):
        if isinstance(obs_shape, int):
            inputs = torch.randn(B, obs_shape)
        else:
            inputs = torch.randn(B, *obs_shape)
        num_quantiles = 48
        model = IQN(obs_shape, act_shape, num_quantiles=num_quantiles, quantile_embedding_size=64)
        outputs = model(inputs)
        print(model)
        assert isinstance(outputs, dict)
        if isinstance(act_shape, int):
            assert outputs['logit'].shape == (B, act_shape)
            assert outputs['q'].shape == (num_quantiles, B, act_shape)
            assert outputs['quantiles'].shape == (B * num_quantiles, 1)
        elif len(act_shape) == 1:
            assert outputs['logit'].shape == (B, *act_shape)
            assert outputs['q'].shape == (num_quantiles, B, *act_shape)
            assert outputs['quantiles'].shape == (B * num_quantiles, 1)
        else:
            for i, s in enumerate(act_shape):
                assert outputs['logit'][i].shape == (B, s)
                assert outputs['q'][i].shape == (num_quantiles, B, s)
                assert outputs['quantiles'][i].shape == (B * num_quantiles, 1)
        self.output_check(model, outputs['logit'])

    @pytest.mark.parametrize('obs_shape, act_shape', args)
    def test_fqf(self, obs_shape, act_shape):
        if isinstance(obs_shape, int):
            inputs = torch.randn(B, obs_shape)
        else:
            inputs = torch.randn(B, *obs_shape)
        num_quantiles = 48
        model = FQF(obs_shape, act_shape, num_quantiles=num_quantiles, quantile_embedding_size=64)
        outputs = model(inputs)
        print(model)
        assert isinstance(outputs, dict)
        if isinstance(act_shape, int):
            assert outputs['logit'].shape == (B, act_shape)
            assert outputs['q'].shape == (B, num_quantiles, act_shape)
            assert outputs['quantiles'].shape == (B, num_quantiles + 1)
            assert outputs['quantiles_hats'].shape == (B, num_quantiles)
            assert outputs['q_tau_i'].shape == (B, num_quantiles - 1, act_shape)
            all_quantiles_proposal = model.head.quantiles_proposal
            all_fqf_fc = model.head.fqf_fc
        elif len(act_shape) == 1:
            assert outputs['logit'].shape == (B, *act_shape)
            assert outputs['q'].shape == (B, num_quantiles, *act_shape)
            assert outputs['quantiles'].shape == (B, num_quantiles + 1)
            assert outputs['quantiles_hats'].shape == (B, num_quantiles)
            assert outputs['q_tau_i'].shape == (B, num_quantiles - 1, *act_shape)
            all_quantiles_proposal = model.head.quantiles_proposal
            all_fqf_fc = model.head.fqf_fc
        else:
            for i, s in enumerate(act_shape):
                assert outputs['logit'][i].shape == (B, s)
                assert outputs['q'][i].shape == (B, num_quantiles, s)
                assert outputs['quantiles'][i].shape == (B, num_quantiles + 1)
                assert outputs['quantiles_hats'][i].shape == (B, num_quantiles)
                assert outputs['q_tau_i'][i].shape == (B, num_quantiles - 1, s)
            all_quantiles_proposal = [h.quantiles_proposal for h in model.head.pred]
            all_fqf_fc = [h.fqf_fc for h in model.head.pred]
        self.output_check(all_quantiles_proposal, outputs['quantiles'])
        for p in model.parameters():
            p.grad = None
        self.output_check(all_fqf_fc, outputs['q'])

    @pytest.mark.parametrize('obs_shape, act_shape', args)
    def test_qrdqn(self, obs_shape, act_shape):
        if isinstance(obs_shape, int):
            inputs = torch.randn(B, obs_shape)
        else:
            inputs = torch.randn(B, *obs_shape)
        model = QRDQN(obs_shape, act_shape, num_quantiles=32)
        outputs = model(inputs)
        assert isinstance(outputs, dict)
        if isinstance(act_shape, int):
            assert outputs['logit'].shape == (B, act_shape)
            assert outputs['q'].shape == (B, act_shape, 32)
            assert outputs['tau'].shape == (B, 32, 1)
        elif len(act_shape) == 1:
            assert outputs['logit'].shape == (B, *act_shape)
            assert outputs['q'].shape == (B, *act_shape, 32)
            assert outputs['tau'].shape == (B, 32, 1)
        else:
            for i, s in enumerate(act_shape):
                assert outputs['logit'][i].shape == (B, s)
                assert outputs['q'][i].shape == (B, s, 32)
                assert outputs['tau'][i].shape == (B, 32, 1)
        self.output_check(model, outputs['logit'])

    @pytest.mark.parametrize('obs_shape, act_shape', args)
    def test_drqn(self, obs_shape, act_shape):
        if isinstance(obs_shape, int):
            inputs = torch.randn(T, B, obs_shape)
        else:
            inputs = torch.randn(T, B, *obs_shape)
        # (num_layer * num_direction, 1, head_hidden_size)
        prev_state = [{k: torch.randn(1, 1, 64) for k in ['h', 'c']} for _ in range(B)]
        model = DRQN(obs_shape, act_shape)
        outputs = model({'obs': inputs, 'prev_state': prev_state}, inference=False)
        assert isinstance(outputs, dict)
        if isinstance(act_shape, int):
            assert outputs['logit'].shape == (T, B, act_shape)
        elif len(act_shape) == 1:
            assert outputs['logit'].shape == (T, B, *act_shape)
        else:
            for i, s in enumerate(act_shape):
                assert outputs['logit'][i].shape == (T, B, s)
        assert len(outputs['next_state']) == B
        assert all([len(t) == 2 for t in outputs['next_state']])
        assert all([t['h'].shape == (1, 1, 64) for t in outputs['next_state']])
        self.output_check(model, outputs['logit'])

    @pytest.mark.parametrize('obs_shape, act_shape', args)
    def test_drqn_inference(self, obs_shape, act_shape):
        if isinstance(obs_shape, int):
            inputs = torch.randn(B, obs_shape)
        else:
            inputs = torch.randn(B, *obs_shape)
        # (num_layer * num_direction, 1, head_hidden_size)
        prev_state = [{k: torch.randn(1, 1, 64) for k in ['h', 'c']} for _ in range(B)]
        model = DRQN(obs_shape, act_shape)
        outputs = model({'obs': inputs, 'prev_state': prev_state}, inference=True)
        assert isinstance(outputs, dict)
        if isinstance(act_shape, int):
            assert outputs['logit'].shape == (B, act_shape)
        elif len(act_shape) == 1:
            assert outputs['logit'].shape == (B, *act_shape)
        else:
            for i, s in enumerate(act_shape):
                assert outputs['logit'][i].shape == (B, s)
        assert len(outputs['next_state']) == B
        assert all([len(t) == 2 for t in outputs['next_state']])
        assert all([t['h'].shape == (1, 1, 64) for t in outputs['next_state']])
        self.output_check(model, outputs['logit'])

    @pytest.mark.parametrize('obs_shape, act_shape', args)
    def test_drqn_res_link(self, obs_shape, act_shape):
        if isinstance(obs_shape, int):
            inputs = torch.randn(T, B, obs_shape)
        else:
            inputs = torch.randn(T, B, *obs_shape)
        # (num_layer * num_direction, 1, head_hidden_size)
        prev_state = [{k: torch.randn(1, 1, 64) for k in ['h', 'c']} for _ in range(B)]
        model = DRQN(obs_shape, act_shape, res_link=True)
        outputs = model({'obs': inputs, 'prev_state': prev_state}, inference=False)
        assert isinstance(outputs, dict)
        if isinstance(act_shape, int):
            assert outputs['logit'].shape == (T, B, act_shape)
        elif len(act_shape) == 1:
            assert outputs['logit'].shape == (T, B, *act_shape)
        else:
            for i, s in enumerate(act_shape):
                assert outputs['logit'][i].shape == (T, B, s)
        assert len(outputs['next_state']) == B
        assert all([len(t) == 2 for t in outputs['next_state']])
        assert all([t['h'].shape == (1, 1, 64) for t in outputs['next_state']])
        self.output_check(model, outputs['logit'])

    @pytest.mark.parametrize('obs_shape, act_shape', args)
    def test_drqn_inference_res_link(self, obs_shape, act_shape):
        if isinstance(obs_shape, int):
            inputs = torch.randn(B, obs_shape)
        else:
            inputs = torch.randn(B, *obs_shape)
        # (num_layer * num_direction, 1, head_hidden_size)
        prev_state = [{k: torch.randn(1, 1, 64) for k in ['h', 'c']} for _ in range(B)]
        model = DRQN(obs_shape, act_shape, res_link=True)
        outputs = model({'obs': inputs, 'prev_state': prev_state}, inference=True)
        assert isinstance(outputs, dict)
        if isinstance(act_shape, int):
            assert outputs['logit'].shape == (B, act_shape)
        elif len(act_shape) == 1:
            assert outputs['logit'].shape == (B, *act_shape)
        else:
            for i, s in enumerate(act_shape):
                assert outputs['logit'][i].shape == (B, s)
        assert len(outputs['next_state']) == B
        assert all([len(t) == 2 for t in outputs['next_state']])
        assert all([t['h'].shape == (1, 1, 64) for t in outputs['next_state']])
        self.output_check(model, outputs['logit'])
