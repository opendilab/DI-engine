from typing import Iterable, Dict, List, Union, Any, Callable
from functools import partial
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.distributed import get_rank
import torch
import torch.nn.functional as F


def zero_pad_sequences(sequences: List[torch.Tensor], side: str = "left", value: int = 0) -> torch.Tensor:
    assert side in ("left", "right")
    max_len = max(seq.size(-1) for seq in sequences)
    padded_sequences = []
    for seq in sequences:
        pad_len = max_len - seq.size(-1)
        padding = (pad_len, 0) if side == "left" else (0, pad_len)
        padded_sequences.append(F.pad(seq, padding, value=value))
    return torch.stack(padded_sequences, dim=0)


class OfflineRLDataset(Dataset):
    """
    Overview:
        PyTorch Dataset for OfflineRL LLM training like KTO and DPO.
    """

    def __init__(
            self,
            dataset: Iterable[Dict],
            tokenizer,
            max_length: int,
            input_key: str = "input",
            extra_input_keys: List[str] = [],
            output_key: str = "output",
            label_key: str = "label",
            apply_chat_template: bool = False,
            tokenizer_chat_template: str = None,
            input_template: str = None,
            num_processors: int = 8,
            parallel_load: bool = True
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.extra_input_keys = extra_input_keys

        if apply_chat_template:
            apply_chat_template = self.tokenizer.apply_chat_template
            if tokenizer_chat_template:
                self.tokenizer.chat_template = tokenizer_chat_template

        # Parallel loading datasets
        if parallel_load:
            preprocess_data_fn = partial(
                self._preprocess_data,
                input_template=input_template,
                input_key=input_key,
                extra_input_keys=extra_input_keys,
                output_key=output_key,
                label_key=label_key,
                apply_chat_template=apply_chat_template
            )
            processed_dataset = dataset.map(
                preprocess_data_fn, remove_columns=dataset.column_names, num_proc=num_processors
            )
            # preprocess function may return None, so filter out the None
            processed_dataset = processed_dataset.filter(lambda x: x["prompt"] is not None)

            self.prompts = processed_dataset["prompt"]
            self.responses = processed_dataset["response"]
            self.labels = processed_dataset["label"]
            self.prompt_ids_lens = processed_dataset["prompt_ids_len"]
            for key in extra_input_keys:
                setattr(self, key, processed_dataset[key])
        else:
            self.prompts = []
            self.responses = []
            self.labels = []
            self.prompt_ids_lens = []
            for key in extra_input_keys:
                setattr(self, key, [])
            for data in tqdm(dataset, desc="Preprocessing data", disable=not get_rank() == 0):
                processed_data = self._preprocess_data(data)
                if processed_data["prompt"] is not None:
                    self.prompts.append(processed_data["prompt"])
                    self.responses.append(processed_data["response"])
                    self.labels.append(processed_data["label"])
                    self.prompt_ids_lens.append(processed_data["prompt_ids_len"])
                    for key in extra_input_keys:
                        getattr(self, key).append(processed_data[key])

    def _preprocess_data(
            self,
            data: Dict[str, Any],
            input_template: str = None,
            input_key: str = "input",
            extra_input_keys: List[str] = [],
            output_key: str = "output",
            label_key: str = "label",
            apply_chat_template: Union[bool, Callable] = False,
    ) -> str:
        label = data[label_key]
        if extra_input_keys:
            extra_inputs = {key: data[key] for key in extra_input_keys}
        else:
            extra_inputs = {}

        if apply_chat_template:
            if output_key:
                prompt = apply_chat_template(data[input_key], tokenize=False, add_generation_prompt=True)
                response = apply_chat_template(data[input_key] + data[output_key], tokenize=False)[len(prompt):]
            else:
                prompt = apply_chat_template(data[input_key][:-1], tokenize=False, add_generation_prompt=True)
                response = apply_chat_template(data[input_key], tokenize=False)[len(prompt):]
        else:
            prompt = data[input_key]
            response = data[output_key]
            if input_template:
                prompt = input_template.format(prompt)

        prompt_token = self.tokenizer(
            prompt,
            max_length=self.max_length,
            # use the batch max length (in `collate_fn`) to pad rather than the global max length
            padding=False,
            truncation=True,
            return_tensors="pt",
            # add special tokens for the prompt in `collate_fn`
            add_special_tokens=False,
        )
        prompt_ids_len = prompt_token["attention_mask"].int().sum().item()

        # filter the sample whose length is greater than max_length (2 for answer length)
        if prompt_ids_len >= self.max_length - 2:
            prompt = None

        return {
            "prompt": prompt,
            "response": response,
            "label": label,
            "prompt_ids_len": prompt_ids_len,
            **extra_inputs
        }

    def __len__(self) -> int:
        """
        Overview:
            Get the length of the dataset.
        Returns:
            - length (int): The length of the dataset.
        """
        return len(self.prompts)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, int]]:
        """
        Overview:
            Get the item at the given index.
        Arguments:
            - idx (int): The index of the item to get.
        Returns:
            - item (Dict[str, Union[torch.Tensor, int]]): The item at the given index.
        """
        if self.extra_input_keys:
            extra_inputs = {key: getattr(self, key)[idx] for key in self.extra_input_keys}
        else:
            extra_inputs = {}
        return {
            "prompt": self.prompts[idx],
            "response": self.responses[idx],
            "label": self.labels[idx],
            "prompt_ids_len": self.prompt_ids_lens[idx],
            **extra_inputs
        }

    def collate_fn(self, item_list: List[Dict[str, Union[torch.Tensor, int]]]):

        def tokenizer(prompt: str, response: str):
            text = (prompt + response).rstrip("\n")
            if not text.endswith(self.tokenizer.eos_token):
                text += " " + self.tokenizer.eos_token
            inputs = self.tokenizer(
                text,
                max_length=self.max_length,
                padding=False,
                truncation=True,
                return_tensors="pt",
                add_special_tokens=False,
            )

            inputs["input_ids"][0][-1] = self.tokenizer.eos_token_id
            inputs["attention_mask"][0][-1] = True
            return inputs["input_ids"], inputs["attention_mask"]

        tot_ids, tot_masks, tot_labels, prompt_ids_lens, tot_extra_inputs = [], [], [], [], {}
        for item in item_list:
            input_ids, attention_mask = tokenizer(item["prompt"], item["response"])
            tot_ids.append(input_ids)
            tot_masks.append(attention_mask)
            tot_labels.append(item["label"])
            prompt_ids_lens.append(item["prompt_ids_len"])
            for key in self.extra_input_keys:
                if key not in tot_extra_inputs:
                    tot_extra_inputs[key] = []
                tot_extra_inputs[key].append(item[key])

        # add unmatched y'| x (used to estimate the KL divergence between policy and reference)
        for idx in range(len(item_list)):
            next_idx = (idx + 1) % len(item_list)
            input_ids, attention_mask = tokenizer(item_list[idx]["prompt"], item_list[next_idx]["response"])
            tot_ids.append(input_ids)
            tot_masks.append(attention_mask)
            tot_labels.append(-1)
            prompt_ids_lens.append(item_list[idx]["prompt_ids_len"])
            for key in self.extra_input_keys:
                if key not in tot_extra_inputs:
                    tot_extra_inputs[key] = []
                tot_extra_inputs[key].append(item_list[idx][key])

        input_ids = zero_pad_sequences(tot_ids, side="right", value=self.tokenizer.pad_token_id)
        attention_mask = zero_pad_sequences(tot_masks, side="right")
        return input_ids, attention_mask, torch.LongTensor(tot_labels), prompt_ids_lens, tot_extra_inputs
