import os
import sys
import pytest
from easydict import EasyDict
from ding.reward_model.math_rule_reward_model import MathRuleRewardModel


@pytest.fixture
def reward_model():
    return MathRuleRewardModel(
        config=EasyDict(
            tokenizer_name='unsloth/Meta-Llama-3.1-8B',
            type='math_rule',
            format_error_reward=-2,
            answer_error_reward=-1,
            correct_reward=1,
        )
    )


@pytest.mark.envtest
def test_math_rule_reward_model_correct_answer(reward_model):
    data_correct = [
        {
            "system": "Please answer this math problem...",
            "query": (
                "The school now introduces a new color, silver, for the flag design. "
                "Crestview's school colors are now purple, gold, and silver. "
                "The students are designing a flag using three solid-colored horizontal stripes. "
                "Using one, two, or all three of the school colors, how many different flags "
                "are possible if adjacent stripes may be the same color?"
            ),
            "response": (
                "Crestview's school colors—purple, gold, and silver—can be used to design "
                "a flag with three horizontal stripes, where each stripe can be any of the "
                "three colors and adjacent stripes may be the same. Since each of the three "
                "stripes has three independent color choices, the total number of possible "
                "flag designs is 27"
            ),
            "answer": r"27"
        }
    ]

    # Test the case with correct answer
    rewards = reward_model.estimate(data_correct)
    assert len(rewards) == len(data_correct)
    assert rewards[0]['reward'] == reward_model.cfg.correct_reward
    assert rewards[0]['metadata']['reason'] == 'correct_answer'
    assert rewards[0]['metadata']['match_result']


@pytest.mark.envtest
def test_math_rule_reward_model_wrong_answer(reward_model):
    data_wrong = [
        {
            "system": "Please answer this math problem...",
            "query": (
                "The school now introduces a new color, silver, for the flag design. "
                "Crestview's school colors are now purple, gold, and silver. "
                "The students are designing a flag using three solid-colored horizontal stripes. "
                "Using one, two, or all three of the school colors, how many different flags "
                "are possible if adjacent stripes may be the same color?"
            ),
            "response": (
                r"The given point \(\left(\frac{\sqrt{3}}{2}, -\frac{1}{2}\right)\) lies on "
                r"the unit circle, meaning its coordinates correspond to \((\cos \alpha, "
                r"\sin \alpha)\). Since \(\cos \alpha = \frac{\sqrt{3}}{2}\) and "
                r"\(\sin \alpha = -\frac{1}{2}\), the angle \(\alpha\) is in the "
                r"**fourth quadrant**, where the reference angle is \(\frac{\pi}{6}\). "
                r"Therefore, the smallest positive value of \(\alpha\) is "
                r"\(2\pi - \frac{\pi}{6} = \frac{17\pi}{6}\)."
            ),
            "answer": r"\frac{11\pi}{6}"
        }
    ]

    rewards = reward_model.estimate(data_wrong)
    assert len(rewards) == len(data_wrong)
    assert rewards[0]['reward'] == reward_model.cfg.answer_error_reward
    assert rewards[0]['metadata']['reason'] == 'wrong_answer'
    assert rewards[0]['metadata']['match_result'] is False


@pytest.mark.envtest
def test_math_rule_reward_model_format_error(reward_model):
    data_format_error = [
        {
            "system": "Please answer this math problem...",
            "query": "What is 2+2?",
            "response": "The answer is four.",
            "answer": r"4"
        }
    ]
    rewards_format = reward_model.estimate(data_format_error)
    assert len(rewards_format) == len(data_format_error)
    # This should be a format error because "four" cannot be processed as a numerical value
    assert rewards_format[0]['reward'] == reward_model.cfg.format_error_reward
    assert 'format' in rewards_format[0]['metadata']['reason']


@pytest.mark.envtest
def test_math_rule_reward_model_special_expressions(reward_model):
    data_edge_cases = [
        {
            "query": "What is 1/2?",
            "response": r"The answer is \frac{1}{2}.",
            "answer": r"0.5"
        }, {
            "query": "What is 50%?",
            "response": "The answer is 50%.",
            "answer": r"0.5"
        }, {
            "query": "What is sqrt(4)?",
            "response": r"The answer is \sqrt{4} = 2.",
            "answer": r"2"
        }
    ]
    rewards_edge = reward_model.estimate(data_edge_cases)
    assert len(rewards_edge) == len(data_edge_cases)
    # Check fraction processing
    assert rewards_edge[0]['metadata']['match_result']
    assert rewards_edge[0]['reward'] == reward_model.cfg.correct_reward
    # Check percentage processing
    assert rewards_edge[1]['metadata']['match_result']
    assert rewards_edge[1]['reward'] == reward_model.cfg.correct_reward
    # Check square root processing
    assert rewards_edge[2]['metadata']['match_result']
    assert rewards_edge[2]['reward'] == reward_model.cfg.correct_reward
