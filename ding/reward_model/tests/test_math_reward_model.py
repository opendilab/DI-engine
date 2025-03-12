import pytest
from easydict import EasyDict
import torch
from unittest.mock import MagicMock

from ding.reward_model import MathRewardModel


@pytest.mark.envtest
def test_math_reward_model():
    # Create configuration
    cfg = EasyDict(dict(
        type='math',
        model_name='Qwen/Qwen2.5-Math-PRM-7B',
    ))

    # Create mock logger and tb_logger
    logger = MagicMock()
    tb_logger = MagicMock()

    # Initialize reward model
    model = MathRewardModel(cfg, "cuda" if torch.cuda.is_available() else "cpu", logger, tb_logger)

    # Test case 1: Simple math problem
    data_simple = [
        {
            "system": "Please reason step by step...",
            "query": "What is 1 + 1?",
            "response": ["First, we have 1", "Then add 1", "Therefore, 1 + 1 = 2"]
        }
    ]

    # Test case 2: Complex word problem
    data_complex = [
        {
            "system": "Please reason step by step, and put your final answer within \\boxed{}.",
            "query": "Sue lives in a fun neighborhood...",
            "response": [
                "To find out how many more pink plastic flamingos...",
                "On Saturday, they take back one third of the flamingos...",
                "On Sunday, the neighbors add another 18 pink plastic flamingos...",
                "To find the difference, subtract the number of white flamingos..."
            ]
        }
    ]

    # Test simple case
    results_simple = model.estimate(data_simple)

    # Verify simple case results
    assert len(results_simple) == 1, "Should return one result"
    assert "reward" in results_simple[0], "Result should contain reward"
    assert "metadata" in results_simple[0], "Result should contain metadata"
    assert "step_rewards" in results_simple[0]["metadata"], "Metadata should contain step_rewards"
    assert len(results_simple[0]["metadata"]["step_rewards"]) == 3, "Should have 3 step rewards"
    assert results_simple[0]["metadata"]["num_steps"] == 3, "Should have 3 steps"

    # Test complex case
    results_complex = model.estimate(data_complex)

    # Verify complex case results
    assert len(results_complex) == 1, "Should return one result"
    assert "reward" in results_complex[0], "Result should contain reward"
    assert "metadata" in results_complex[0], "Result should contain metadata"
    assert "step_rewards" in results_complex[0]["metadata"], "Metadata should contain step_rewards"
    assert len(results_complex[0]["metadata"]["step_rewards"]) == 4, "Should have 4 step rewards"
    assert results_complex[0]["metadata"]["num_steps"] == 4, "Should have 4 steps"

    # Verify reward value ranges
    for result in results_simple + results_complex:
        assert 0 <= result["reward"] <= 1, "Reward should be between 0 and 1"
        for step_reward in result["metadata"]["step_rewards"]:
            assert 0 <= step_reward <= 1, "Step rewards should be between 0 and 1"

    # Test batch processing functionality
    batch_data = data_simple + data_complex
    batch_results = model.estimate(batch_data)
    assert len(batch_results) == 2, "Should return two results for batch processing"

    # Print detailed information for debugging
    print("\nSimple problem results:")
    print(f"Final reward: {results_simple[0]['reward']}")
    print(f"Step rewards: {results_simple[0]['metadata']['step_rewards']}")

    print("\nComplex problem results:")
    print(f"Final reward: {results_complex[0]['reward']}")
    print(f"Step rewards: {results_complex[0]['metadata']['step_rewards']}")
