from easydict import EasyDict
import pytest
from dizoo.tabmwp.envs.tabmwp_env import TabMWP


@pytest.mark.envtest
class TestSokoban:

    def test_tabmwp(self):
        config = dict(
            cand_number=20,
            train_number=100,
            engine='text-davinci-002',
            temperature=0.,
            max_tokens=512,
            top_p=1.,
            frequency_penalty=0.,
            presence_penalty=0.,
            option_inds=["A", "B", "C", "D", "E", "F"],
            api_key='',
        )
        config = EasyDict(config)
        env = TabMWP(config)
        env.seed(0)
        env.close()
