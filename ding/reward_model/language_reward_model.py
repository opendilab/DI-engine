import torch
from transformers.models.llama.modeling_llama import LlamaForCausalLM


class LlamaRewardModel(LlamaForCausalLM):

    def __init__(self, config, opt, tokenizer):
        super().__init__(config)
        self.opt = opt
        self.tokenizer = tokenizer
        self.reward_head = torch.nn.Linear(config.hidden_size, 1, bias=False)

    def forward(self, decoder_input, only_last=True):
        attention_mask = decoder_input.ne(self.tokenizer.pad_token_id)
        output = self.model.forward(
            input_ids=decoder_input, attention_mask=attention_mask, return_dict=True, use_cache=False
        )

        if only_last:
            logits = self.reward_head(output.last_hidden_state[:, -1, :]).squeeze(-1)
        else:
            logits = self.reward_head(output.last_hidden_state).squeeze(-1)

        return (logits, )
