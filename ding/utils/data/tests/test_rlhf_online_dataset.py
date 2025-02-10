import pytest
from datasets import load_dataset
from ding.utils.data import OnlineRLDataset
from transformers import AutoTokenizer
IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
IMG_START_TOKEN = '<img>'
IMG_END_TOKEN = '</img>'
IMG_CONTEXT_NUM = 10  # user-defined number of image patches in the context
@pytest.fixture
def dataset():
    # Load the dataset
    hf_dataset = load_dataset("MMInstruction/VL-RewardBench",split='test')
    hf_dataset0 = hf_dataset.map(
        lambda x: {
            "query": f"{IMG_START_TOKEN}{IMG_CONTEXT_TOKEN * IMG_CONTEXT_NUM}{IMG_END_TOKEN}\n{x['query']}",
            "image": x["image"],
        }
    )
    # shuffle the dataset
    hf_dataset = hf_dataset0.shuffle(seed=42)
    print(hf_dataset)
    return hf_dataset


@pytest.fixture
def tokenizer():
    return AutoTokenizer.from_pretrained("OpenGVLab/InternVL2_5-4B")


@pytest.mark.unittest
def test_onlinerl_dataset_initialization(dataset, tokenizer):
    # Initialize OnlineRLDataset
    online_rl_dataset = OnlineRLDataset(
        dataset=dataset, tokenizer=tokenizer, input_key="query",
        extra_input_keys=["image"], apply_chat_template=True
    )
    # Check if the dataset is initialized correctly
    assert len(online_rl_dataset) == len(dataset)


@pytest.mark.unittest
def test_onlinerl_dataset_getitem(dataset, tokenizer):
    # Initialize OnlineRLDataset
    online_rl_dataset = OnlineRLDataset(
        dataset=dataset, tokenizer=tokenizer, input_key="query",
        extra_input_keys=["image"], apply_chat_template=True
    )
    # Check if __getitem__ returns the expected formatted prompt
    item = online_rl_dataset[0]
    print(item)
    assert "prompt" in item
    assert "multi_modal_data" in item
    assert "image" in item['multi_modal_data']
    assert isinstance(item['prompt'],str)
