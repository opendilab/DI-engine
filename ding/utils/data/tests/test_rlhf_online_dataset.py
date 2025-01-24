import pytest
from datasets import load_dataset
from transformers import AutoTokenizer
from rl.data.onlinerl_dataset import OnlineRLDataset


@pytest.fixture
def dataset():
    # Load the dataset
    hf_dataset = load_dataset("cat-searcher/minif2f-lean4")['validation']
    print(hf_dataset)
    return hf_dataset


@pytest.fixture
def tokenizer():
    return AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-7B")


@pytest.mark.unittest
def test_onlinerl_dataset_initialization(dataset, tokenizer):
    # Initialize OnlineRLDataset
    online_rl_dataset = OnlineRLDataset(
        dataset=dataset, tokenizer=tokenizer, input_key="formal_statement", apply_chat_template=True
    )
    # Check if the dataset is initialized correctly
    assert len(online_rl_dataset) == len(dataset)


@pytest.mark.unittest
def test_onlinerl_dataset_getitem(dataset, tokenizer):
    # Initialize OnlineRLDataset
    online_rl_dataset = OnlineRLDataset(
        dataset=dataset, tokenizer=tokenizer, input_key="formal_statement", apply_chat_template=True
    )
    # Check if __getitem__ returns the expected formatted prompt
    item = online_rl_dataset[0]
    print(item)
    assert isinstance(item, str)
