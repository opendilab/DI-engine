import pytest
from easydict import EasyDict
import torch
from ding.reward_model import MultiModalRewardModel
from unittest.mock import MagicMock
import os


@pytest.fixture
def reward_model():
    # Create configuration
    cfg = EasyDict(dict(
        type='multi_modal',
        model_name='internlm/internlm-xcomposer2d5-7b-reward',
        hd_num=9,
    ))

    # Create mock logger and tb_logger
    logger = MagicMock()
    tb_logger = MagicMock()

    # Initialize reward model
    model = MultiModalRewardModel(cfg, "cuda" if torch.cuda.is_available() else "cpu", logger, tb_logger)
    return model


@pytest.fixture
def test_data():
    # Shared test data
    chats = [
        [  # chat_1
            {"role": "user", "content": 'I want to buy a car from the input image, '
                                        'analyze the advantages and weaknesses.'},
            {"role": "assistant", "content": "The car in the image is a Mercedes-Benz G-Class..."}
        ],
        [  # chat_2
            {"role": "user", "content": 'I want to buy a car from the input image, '
                                        'analyze the advantages and weaknesses.'},
            {"role": "assistant", "content": "Based on the image, it appears to be a Ferrari F8 Tributo..."}
        ]
    ]

    images = ['./examples/cars1.jpg']

    return {'chats': chats, 'images': images, 'hd_num': 9}


@pytest.mark.envtest
def test_single_score(reward_model, test_data):
    """Test single chat scoring"""
    data = [{'chat': test_data['chats'][0]}]

    results = reward_model.estimate(data, test_data['images'], output_mode='score')
    print(f"Single score results: {results}")

    assert len(results) == 1
    assert 'reward' in results[0]
    assert isinstance(results[0]['reward'], float)
    assert results[0]['metadata']['mode'] == 'score'
    assert results[0]['metadata']['chat_idx'] == 0


@pytest.mark.envtest
def test_multiple_scores(reward_model, test_data):
    """Test multiple chats scoring"""
    data = [{'chat': test_data['chats'][0]}, {'chat': test_data['chats'][1]}]

    results = reward_model.estimate(data, test_data['images'], output_mode='score')
    print(f"Multiple scores results: {results}")

    assert len(results) == 2
    assert all('reward' in r for r in results)
    assert all(isinstance(r['reward'], float) for r in results)
    assert all(r['metadata']['mode'] == 'score' for r in results)


@pytest.mark.envtest
def test_rank(reward_model, test_data):
    """Test ranking functionality"""
    data = [{'chat': test_data['chats'][0]}, {'chat': test_data['chats'][1]}]

    results = reward_model.estimate(data, test_data['images'], output_mode='rank')
    print(f"Rank results: {results}")

    assert len(results) == 2
    assert all('rank' in r for r in results)
    assert set(r['rank'] for r in results) == {0, 1}


@pytest.mark.envtest
def test_compare(reward_model, test_data):
    """Test comparison functionality"""
    data = [{'chat': test_data['chats'][0]}, {'chat': test_data['chats'][1]}]

    results = reward_model.estimate(data, test_data['images'], output_mode='compare')
    print(f"Compare results: {results}")

    assert len(results) == 2
    assert sum(r['reward'] for r in results) == 1.0
    assert all(r['metadata']['mode'] == 'compare' for r in results)


@pytest.mark.envtest
def test_default_parameters(reward_model, test_data):
    """Test default parameters"""
    data = [{'chat': test_data['chats'][0]}]

    # Test without specifying optional parameters
    results = reward_model.estimate(data, test_data['images'])

    assert len(results) == 1
    assert 'reward' in results[0]
    assert results[0]['metadata']['mode'] == 'score'


@pytest.mark.envtest
def test_error_handling(reward_model, test_data):
    """Test error handling"""
    with pytest.raises(Exception):
        # Test invalid input format
        reward_model.model.get_score(None, test_data['image'], hd_num=test_data['hd_num'])
