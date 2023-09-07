import torch

from ding.utils import MODEL_REGISTRY
from torch import nn
try:
    from transformers import AutoTokenizer, AutoModelForTokenClassification
except ImportError:
    from ditk import logging
    logging.warning("not found transformer, please install it using: pip install transformers")


@MODEL_REGISTRY.register('language_transformer')
class LanguageTransformer(nn.Module):

    def __init__(
            self,
            model_name: str = "bert-base-uncased",
            add_linear: bool = False,
            embedding_size: int = 128,
            freeze_encoder: bool = True
    ) -> None:
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)

        # Freeze transformer encoder and only train the linear layer
        if freeze_encoder:
            for param in self.model.parameters():
                param.requires_grad = False

        if add_linear:
            # Add an additional small, adjustable linear layer on top of BERT tuned through RL
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

    def forward(self, train_samples: list, candidate_samples: list) -> dict:
        prompt_embedding = self._calc_embedding(train_samples)
        cands_embedding = self._calc_embedding(candidate_samples)
        scores = torch.mm(prompt_embedding, cands_embedding.t())
        return {'dist': torch.distributions.Categorical(logits=scores), 'logit': scores}
