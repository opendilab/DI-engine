from typing import List, Dict
import torch
from torch import nn

try:
    from transformers import AutoTokenizer, AutoModelForTokenClassification
except ImportError:
    from ditk import logging
    logging.warning("not found transformer, please install it using: pip install transformers")
from ding.utils import MODEL_REGISTRY


@MODEL_REGISTRY.register('language_transformer')
class LanguageTransformer(nn.Module):
    """
    Overview:
        The LanguageTransformer network. Download a pre-trained language model and add head on it.
    Interfaces:
        ``__init__``, ``forward``
    """

    def __init__(
            self,
            model_name: str = "bert-base-uncased",
            add_linear: bool = False,
            embedding_size: int = 128,
            freeze_encoder: bool = True
    ) -> None:
        """
        Overview:
            Init the LanguageTransformer Model according to input arguments.
        Arguments:
            - model_name (:obj:`str`): The base language model name in huggingface, such as "bert-base-uncased".
            - add_linear (:obj:`bool`): Whether to add a linear layer on the top of language model, defaults to be \
            ``False``.
            - embedding_size (:obj:`int`): The embedding size of the added linear layer, such as 128.
            - freeze_encoder (:obj:`bool`): Whether to freeze the encoder language model while training, \
            defaults to be ``True``.
        """
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)

        # Freeze transformer encoder and only train the linear layer
        if freeze_encoder:
            for param in self.model.parameters():
                param.requires_grad = False

        if add_linear:
            # Add a small, adjustable linear layer on top of language model tuned through RL
            self.embedding_size = embedding_size
            self.linear = nn.Linear(
                self.model.config.hidden_size, embedding_size
            )  # 768 for bert-base-uncased, distilbert-base-uncased
        else:
            self.linear = None

    def _calc_embedding(self, x: list) -> torch.Tensor:
        # ``truncation=True`` means that if the length of the prompt exceed the ``max_length`` of the tokenizer,
        # the exceeded part will be truncated. ``padding=True`` means that if the length of the prompt does not reach
        # the ``max_length``, the latter part will be padded. These settings ensure the length of encoded tokens is
        # exactly ``max_length``, which can enable batch-wise computing.
        input = self.tokenizer(x, truncation=True, padding=True, return_tensors="pt").to(self.model.device)
        output = self.model(**input, output_hidden_states=True)
        # Get last layer hidden states
        last_hidden_states = output.hidden_states[-1]
        # Get [CLS] hidden states
        sentence_embedding = last_hidden_states[:, 0, :]  # len(input_list) x hidden_size

        if self.linear:
            sentence_embedding = self.linear(sentence_embedding)  # len(input_list) x embedding_size

        return sentence_embedding

    def forward(self, train_samples: List[str], candidate_samples: List[str]) -> Dict:
        """
        Overview:
            LanguageTransformer forward computation graph, input two lists of strings and predict their matching scores.
        Arguments:
            - train_samples (:obj:`List[str]`): One list of strings.
            - candidate_samples (:obj:`List[str]`): The other list of strings to calculate the matching scores.
        Returns:
            - output (:obj:`Dict`): Output dict data, including the logit of matching scores and the \
            corresponding ``torch.distributions.Categorical`` object.

        Examples:
            >>> test_pids = [1]
            >>> cand_pids = [0, 2, 4]
            >>> problems = [ \
                "This is problem 0", "This is the first question", "Second problem is here", "Another problem", \
                "This is the last problem" \
            ]
            >>> ctxt_list = [problems[pid] for pid in test_pids]
            >>> cands_list = [problems[pid] for pid in cand_pids]
            >>> model = LanguageTransformer(model_name="bert-base-uncased", add_linear=True, embedding_size=256)
            >>> scores = model(ctxt_list, cands_list)
            >>> assert scores.shape == (1, 3)
        """
        prompt_embedding = self._calc_embedding(train_samples)
        cands_embedding = self._calc_embedding(candidate_samples)
        scores = torch.mm(prompt_embedding, cands_embedding.t())
        return {'dist': torch.distributions.Categorical(logits=scores), 'logit': scores}
