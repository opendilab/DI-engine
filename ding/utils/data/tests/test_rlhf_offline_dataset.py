import pytest
from datasets import load_dataset, concatenate_datasets
from ding.utils.data import OfflineRLDataset
from transformers import AutoTokenizer


@pytest.fixture
def dataset():
    # Load a sample dataset
    hf_dataset = load_dataset("MMInstruction/VL-RewardBench", split='test')
    # split pair data into two separate datasets
    hf_dataset_1 = hf_dataset.map(
        lambda x: {
            "prompt": x["query"],
            "response": x["response"][0],
            'human_ranking': x["human_ranking"][0]
        }
    )
    hf_dataset_2 = hf_dataset.map(
        lambda x: {
            "prompt": x["query"],
            "response": x["response"][1],
            'human_ranking': x["human_ranking"][0]
        }
    )
    # combine two datasets
    hf_dataset = concatenate_datasets([hf_dataset_1, hf_dataset_2])
    # shuffle the dataset
    hf_dataset = hf_dataset.shuffle(seed=42)
    return hf_dataset


@pytest.fixture
def tokenizer():
    # Load a tokenizer
    return AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-7B")


@pytest.mark.unittest
def test_offline_rl_dataset_initialization(dataset, tokenizer):
    # Test the initialization of the OfflineRLDataset
    offline_dataset = OfflineRLDataset(
        dataset=dataset,
        tokenizer=tokenizer,
        max_length=1024,
        input_key="query",
        output_key="response",
        label_key="human_ranking"
    )
    assert len(offline_dataset) == len(dataset)
    offline_dataset = OfflineRLDataset(
        dataset=dataset,
        tokenizer=tokenizer,
        max_length=256,
        input_key="query",
        output_key="response",
        label_key="human_ranking"
    )
    # lower max_length will filter out some samples
    assert len(offline_dataset) < len(dataset)


@pytest.mark.unittest
def test_offline_rl_dataset_item_retrieval(dataset, tokenizer):
    # Test retrieving an item from the OfflineRLDataset
    offline_dataset = OfflineRLDataset(
        dataset=dataset,
        tokenizer=tokenizer,
        max_length=256,
        input_key="query",
        output_key="response",
        label_key="human_ranking"
    )
    item = offline_dataset[0]
    assert "prompt" in item
    assert "response" in item
    assert "label" in item
    assert "prompt_ids_len" in item
    print(item)


@pytest.mark.unittest
def test_offline_rl_dataset_collate_fn(dataset, tokenizer):
    # Test the collate function of the OfflineRLDataset
    offline_dataset = OfflineRLDataset(
        dataset=dataset,
        tokenizer=tokenizer,
        max_length=256,
        input_key="query",
        output_key="response",
        label_key="human_ranking"
    )
    B = 10
    item_list = [offline_dataset[i] for i in range(B)]
    input_ids, attention_mask, labels, prompt_ids_lens = offline_dataset.collate_fn(item_list)
    assert input_ids.size(0) == len(item_list) * 2  # because of the unmatched y'| x
    assert attention_mask.size(0) == len(item_list) * 2
    assert labels.size(0) == len(item_list) * 2
    assert len(prompt_ids_lens) == len(item_list) * 2
