from typing import Any, Dict, Union, Callable, Iterable
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.distributed import get_rank
from transformers import AutoTokenizer


class OnlineRLDataset(Dataset):
    """
    Overview:
        PyTorch Dataset for OnlineRL LLM training like PPO.
        This dataset only supports pure text input now.
    """

    def __init__(
            self,
            dataset: Iterable[Dict],
            tokenizer: AutoTokenizer,
            input_key: str = "input",
            apply_chat_template: bool = False,
            input_template: str = None,
    ) -> None:
        """
        Overview:
            Initialize the OnlineRLDataset.
        Arguments:
            - dataset (torch.utils.data.Dataset): The dataset to preprocess.
            - tokenizer (AutoTokenizer): The tokenizer to preprocess the data.
            - input_key (str): The key of the input data, default is "input".
            - apply_chat_template (bool): Whether to apply the chat template, default is False.
            - input_template (str): The template to format the data.
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.input_template = input_template

        if apply_chat_template:
            apply_chat_template = self.tokenizer.apply_chat_template

        self.prompts = []
        try:
            rank = get_rank()
        except ValueError:  # not initialized yet, which is the case in unit test
            rank = 0
        for data in tqdm(dataset, desc="Preprocessing data", disable=not rank == 0):
            prompt = self._preprocess_data(data, input_template, input_key, apply_chat_template)
            self.prompts.append(prompt)

    def __len__(self) -> int:
        """
        Overview:
            Get the length of the dataset.
        Returns:
            - length (int): The length of the dataset.
        """
        return len(self.prompts)

    def __getitem__(self, idx: int) -> str:
        """
        Overview:
            Get the item at the given index.
        Arguments:
            - idx (int): The index of the item to get.
        Returns:
            - item (str): The item at the given index.
        """
        return self.prompts[idx]

    def _preprocess_data(
            self,
            data: Dict[str, Any],
            input_template: str = None,
            input_key: str = "input",
            apply_chat_template: Union[bool, Callable] = False,
    ) -> str:
        """
        Overview:
            Preprocess the data to get the formatted prompt.
        Arguments:
            - data (Dict[str, Any]): The data to preprocess.
            - input_template (str): The template to format the data.
            - input_key (str): The key of the input data.
            - apply_chat_template (Union[bool, Callable]): Controls chat template application. If True, uses the \
                tokenizer's default template. If a Callable is provided, uses that function to apply the template \
                (typically tokenizer.apply_chat_template).
        Returns:
            - prompt (str): The formatted prompt.
        """
        if apply_chat_template:
            chat = data[input_key]
            if isinstance(chat, str):
                chat = [{"role": "user", "content": chat}]
            assert isinstance(chat, list) and all(isinstance(t, dict) for t in chat), "chat must be a list of dict"
            prompt = apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        else:
            prompt = data[input_key]
            if input_template:
                prompt = input_template.format(prompt)
        return prompt
