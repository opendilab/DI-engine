import pytest
import torch

from ding.model.template.nlp_pretrained_model import NLPPretrainedModel


@pytest.mark.unittest
class TestNLPPretrainedModel:
    def check_model(self):
        test_pids = [1]
        cand_pids = [0, 2, 4]
        problems = [
            "This is problem 0", "This is the first question", "Second problem is here", "Another problem",
            "This is the last problem"
        ]
        ctxt_list = [problems[pid] for pid in test_pids]
        cands_list = [problems[pid] for pid in cand_pids]

        model = NLPPretrainedModel(model_config="bert-base-uncased", add_linear=True, embedding_size=256)
        cands_embedding = model(cands_list)
        assert cands_embedding.shape == (3, 256)
        ctxt_embedding = model(ctxt_list)
        assert ctxt_embedding.shape == (1, 256)

        scores = torch.mm(ctxt_embedding, cands_embedding.t())
        assert scores.shape == (1, 3)
