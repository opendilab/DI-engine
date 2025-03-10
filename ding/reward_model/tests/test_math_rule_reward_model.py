import pytest
from easydict import EasyDict

from ding.reward_model import MathRuleRewardModel


@pytest.mark.envtest
def test_math_rule_reward_model():
    reward_model = MathRuleRewardModel(
        config=EasyDict(
            dataset_name='RUC-AIBOX/STILL-3-Preview-RL-Data',
            tokenizer_name='unsloth/Meta-Llama-3.1-8B',
        )
    )

    data = [
        "The school now introduces a new color, silver, for the flag design. Crestview's school colors are now purple, gold, and silver. The students are designing a flag using three solid-colored horizontal stripes. Using one, two, or all three of the school colors, how many different flags are possible if adjacent stripes may be the same color?",  # noqa
    ]
    rewards = reward_model.estimate(data)
    assert len(rewards) == len(data)
