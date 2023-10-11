import pytest

from ding.model.template.language_transformer import LanguageTransformer


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

        model = LanguageTransformer(model_name="bert-base-uncased", add_linear=True, embedding_size=256)
        scores = model(ctxt_list, cands_list)
        assert scores.shape == (1, 3)

        model = LanguageTransformer(model_name="bert-base-uncased", add_linear=False, embedding_size=256)
        scores = model(ctxt_list, cands_list)
        assert scores.shape == (1, 3)
