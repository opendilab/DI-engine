import json
import os
from typing import List, Dict, Any, Tuple
import warnings

from transformers.models.llama.tokenization_llama import LlamaTokenizer
from torch.utils.data.dataset import IterableDataset
import torch
import random


# Prefix of human sentence and assistant sentence.
HUMAN_PROMPT = "Human:"
ASSISTANT_PROMPT = "Assistant:"


def strip_pad_token_id(tokenizer: LlamaTokenizer, seq: List[int]):
    """
    Overview:
        Remove ``pad_token_id`` in a sequence.
    """
    return [tok for tok in seq if tok != tokenizer.pad_token_id]


def concat_context_and_response(
        tokenizer: LlamaTokenizer,
        context: List[List[int]],
        responses: List[List[Tuple[float, List[int]]]]
):
    """
    Overview:
        Given the batched input prompts and responses, concatenate them together.
    """
    assert len(context) == len(responses), f'Size not match: {len(context)} and {len(responses)}'

    total_context, total_response = [], []
    total_context_mask, total_response_mask = [], []
    for _context, _response in zip(context, responses):
        # Each ``_context`` is a single input prompt.
        _context = strip_pad_token_id(tokenizer, _context)
        for _, resp in _response:
            # Each ``resp`` is a single response.
            resp = strip_pad_token_id(tokenizer, resp)
            if resp[-1] != tokenizer.eos_token_id:
                warnings.warn(
                    f'Generated response is too long: {tokenizer.decode(_context + resp, skip_special_tokens=False)}')

            total_context.append(_context.copy())
            total_context_mask.append([0] * len(_context))
            total_response.append(resp)
            total_response_mask.append([1] * len(resp))

    total_gene_samples_vec = [c + r for c, r in zip(total_context, total_response)]
    total_gene_samples_mask = [c + r for c, r in zip(total_context_mask, total_response_mask)]
    return total_gene_samples_mask, total_gene_samples_vec


def pad_sequences(
        seqs: List[List[int]],
        pad_value: int,
        padding: str = 'right'):
    """
    Overview:
        Padding sequence to the same length
    """
    max_len = max(len(seq) for seq in seqs)
    if padding == 'right':
        padded_seqs = [seq + [pad_value] * (max_len - len(seq)) for seq in seqs]
    elif padding == 'left':
        padded_seqs = [[pad_value] * (max_len - len(seq)) + seq for seq in seqs]
    else:
        raise ValueError
    return padded_seqs


def get_special_prompt(i: int):
    return HUMAN_PROMPT if i % 2 == 0 else ASSISTANT_PROMPT


def get_model_prompt(context: List[str], eos_token="</s>"):
    human_prompt, assistant_prompt = HUMAN_PROMPT, ASSISTANT_PROMPT
    if context[-1].startswith(human_prompt):
        end_prompt = assistant_prompt
    elif context[-1].startswith(assistant_prompt):
        end_prompt = human_prompt
    else:
        raise ValueError

    context = eos_token.join(context)
    return f'{context}{eos_token}{end_prompt}'


def get_tokenizer(path: str):
    """
    Overview:
        Return the pretrained tokenizer using the given path.
    """
    tokenizer = LlamaTokenizer.from_pretrained(path, trust_remote_code=True)
    tokenizer.bos_token = '<s>'
    tokenizer.eos_token = '</s>'
    tokenizer.pad_token = '<unk>'
    tokenizer.pad_token_id = 0
    tokenizer.unk_token = tokenizer.pad_token
    tokenizer.unk_token_id = tokenizer.pad_token_id

    return tokenizer


class OnlyPromptDataset(IterableDataset):
    """
    Overview:
        Dataset that only contains the prompts of the raw data (no answer).
    """
    def __init__(
            self,
            data_path: os.PathLike,
            tokenizer,
            batch_size: int,
            maxlen_prompt: int,
            maxlen_res: int,
            mode: str = 'train',
    ) -> None:
        super().__init__()
        self.mode = mode
        self.tokenizer = tokenizer
        self.maxlen_prompt = maxlen_prompt
        self.maxlen_res = maxlen_res
        self.batch_size = batch_size

        # Load data.
        self.data = []
        files = sorted([file for file in os.listdir(data_path) if file.endswith(f'{mode}.json')])
        for file in files:
            file_path = os.path.join(data_path, file)
            tmp_data = []
            try:
                tmp_data = self.load_data(file_path)
            except Exception as e:
                pass
            self.data.extend(tmp_data)

        # Set the length of this dataset.
        self.size = len(self.data)

    def __len__(self):
        return self.size

    def load_data(self, file_path: str):
        """
        Overview:
            Load raw data from given file_path.
        """
        with open(file_path, 'r') as f:
            data: List[List[str]] = json.load(f)

        output: List[List[str]] = [sample for sample in data if all(sample)]
        del data

        return output

    def final_generator(self):
        data_generator = self.batch_generator()
        for batch_samples in data_generator:
            batch = self.batchify(batch_samples)
            yield batch

    def __iter__(self):
        return self.final_generator()

    def format(self, sample: List[str]) -> Dict[str, Any]:
        """
        Overview:
            Convert one data sample in to string.
        """
        context = sample
        context = [get_special_prompt(i + (len(context) + 1) % 2) + s for i, s in enumerate(context)]
        context_vec = self.tokenizer.encode(get_model_prompt(context, self.tokenizer.eos_token),
                                            add_special_tokens=True)

        # truncate to max_len
        while len(context_vec) > self.maxlen_prompt - self.maxlen_res and len(context) > 1:
            context = context[1:]
            context_vec = self.tokenizer.encode(get_model_prompt(context, self.tokenizer.eos_token),
                                                add_special_tokens=True)

        output = {
            'text': self.tokenizer.decode(context_vec, skip_special_tokens=False),
            'text_vec': context_vec
        }

        return output

    def batchify(self, batch_samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Overview:
            Batchify a list of ids by padding their shape to be the same.
        """
        batch_text_vec = torch.tensor(pad_sequences(
            [sample['text_vec'] for sample in batch_samples], pad_value=self.tokenizer.pad_token_id, padding='left'
        ), dtype=torch.long)
        return {
            'text_vec': batch_text_vec,
            'text': [sample['text'] for sample in batch_samples]
        }

    def sample_generator(self):
        """
        Overview:
            Generate a single data sample.
        """
        random.seed(None)
        if self.mode == 'train':
            random.shuffle(self.data)

        for sample in self.data:
            yield self.format(sample)

    def _batch_generator(self):
        """
        Overview:
            Generate a batch of samples.
        """
        batch = []
        # Generate a sample.
        for sample in self.sample_generator():
            sample_len = len(sample['text_vec'])
            if sample_len > self.maxlen_prompt:
                continue

            batch.append(sample)
            if len(batch) >= self.batch_size:
                yield batch[:self.batch_size]
                batch = batch[self.batch_size:]
        if batch:
            yield batch

    def batch_generator(self):
        while True:
            for batch in self._batch_generator():
                if len(batch) == self.batch_size:
                    yield batch
            if self.mode != 'train':
                break
