from typing import Dict
import torch
import torch.nn as nn
try:
    from transformers import LlamaTokenizer
    from transformers.models.llama.modeling_llama import LlamaForCausalLM
except ImportError:
    from ditk import logging
    logging.warning("Not found transformer, please install it using: pip install transformers")

from ding.model.common import top_p_logits
from ding.reward_model import LlamaRewardModel
from ding.utils import MODEL_REGISTRY


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


class Llama(LlamaForCausalLM):

    def __init__(self, config, opt, tokenizer):
        super().__init__(config)
        self.opt = opt
        self.tokenizer = tokenizer

    def forward(self, decoder_input, incr_state=None, is_train=True):

        attention_mask = decoder_input.ne(self.tokenizer.pad_token_id)
        if incr_state is not None:
            decoder_input = decoder_input[:, -1:]

        output = super().forward(
            input_ids=decoder_input,
            attention_mask=attention_mask,
            past_key_values=incr_state,
            return_dict=True,
            use_cache=not is_train
        )

        logits = output.logits
        new_incr_states = output.past_key_values

        return logits, new_incr_states

    @torch.no_grad()
    def generate(self, batch, **kwargs):
        """
        Generate response
        """
        maxlen_res = kwargs.pop('maxlen_res', self.opt.maxlen_res)
        temperature = kwargs.pop('temperature', self.opt.temperature)
        repetition_penalty = kwargs.pop('repetition_penalty', self.opt.repetition_penalty)
        topp = kwargs.pop('topp', self.opt.topp)

        decoder_input: torch.LongTensor = batch  # (bsz, ...)
        assert decoder_input[:, -1].ne(
            self.tokenizer.pad_token_id
        ).all(), 'Last token should not be a padding token (you can use left padding instead).'

        dev = decoder_input.device
        bsz = decoder_input.size(0)

        scores = torch.zeros((bsz, ), device=dev, dtype=torch.float16)
        done = torch.zeros((bsz, ), device=dev).to(torch.bool)

        inds = torch.arange(bsz).to(dev).unsqueeze(1).view(-1)
        decoder_input = torch.index_select(decoder_input, 0, inds)
        init_length = decoder_input.size(1)

        incr_state = None
        for _token in range(maxlen_res):
            if done.all():
                break
            score, incr_state, *_ = self.forward(decoder_input, incr_state, is_train=False)
            score = score.half()

            # now score is bs, len, vocab_size
            score = score[:, -1, :]

            # calculate repetition penalty
            if repetition_penalty > 1.:
                penalty_tokens = decoder_input[:, init_length:]
                penalty_scores = torch.gather(score, dim=1, index=penalty_tokens)
                penalty_scores = torch.where(
                    penalty_scores < 0., penalty_scores * repetition_penalty, penalty_scores / repetition_penalty
                )
                score = score.scatter_(dim=1, index=penalty_tokens, src=penalty_scores)

            # nucleus sampling
            score = torch.softmax(score.div(temperature), dim=-1)
            probs = top_p_logits(score, topp=topp, filter_value=0)
            tok_ids = torch.multinomial(probs, 1)[:, 0]
            hyp_ids = torch.arange(probs.size(0), device=dev)
            scores = scores + probs[hyp_ids, tok_ids].log() * ~done

            tok_ids = torch.where(done, self.tokenizer.pad_token_id, tok_ids)
            decoder_input = torch.cat((decoder_input, tok_ids.unsqueeze(-1)), dim=-1)
            done = done | tok_ids.eq(self.tokenizer.eos_token_id)

            incr_state = self._reorder_cache(incr_state, hyp_ids)

        # get all finalized candidates for each sample
        decoder_input = decoder_input[:, init_length:]
        decoder_input = decoder_input.view(bsz, -1)
        scores = scores.view(bsz, )

        lengths = decoder_input.ne(self.tokenizer.pad_token_id).sum(dim=-1)

        length_penalty = torch.pow(lengths, 1.0)
        scores /= length_penalty

        preds_scores = []
        for i in range(bsz):
            seq: torch.LongTensor = decoder_input[i, :lengths[i, ]]
            res_scores = (float(scores[i, ]), seq.tolist())
            preds_scores.append([res_scores])

        best_preds_scores = [preds[0] for preds in preds_scores]
        return best_preds_scores, preds_scores


@MODEL_REGISTRY.register('llamavac')
class LlamaVAC(nn.Module):
    """
    Overview:
        The neural network and computation graph of Llama VAC. The actor and critic of this model are respectively \
        a Llama Pretrained Model.
    Interfaces:
        ``__init__``, ``forward``.
    """
    mode = ['compute_actor', 'compute_critic', 'compute_actor_critic']

    def __init__(
            self,
            actor_path: str,
            critic_path: str,
            tokenizer_path: str,
            opt: Dict,
            enable_checkpointing: bool = True
    ) -> None:
        """
        Overview:
            Initialize the ``LlamaVAC`` model according to arguments.
        Arguments:
            - actor_path (:obj:`str`): Pretrained model path for actor.
            - critic_path (:obj:`str`): Pretrained model path for critic.
            - opt (:obj:`Dict`): Options for this model.
        """
        super(LlamaVAC, self).__init__()
        tokenizer = get_tokenizer(tokenizer_path)

        self.actor = Llama.from_pretrained(actor_path, opt=opt, tokenizer=tokenizer, torch_dtype=torch.bfloat16)

        self.critic = LlamaRewardModel.from_pretrained(critic_path, tokenizer=tokenizer, torch_dtype=torch.bfloat16)

        if enable_checkpointing:
            self.actor.gradient_checkpointing_enable()
            self.critic.gradient_checkpointing_enable()

    def forward(self, x: torch.Tensor, mode: str) -> Dict:
        assert mode in self.mode, "not support forward mode: {}/{}".format(mode, self.mode)
        return getattr(self, mode)(x)

    def compute_actor(self, x):
        policy_output = self.actor(decoder_input=x)
        policy_logit, *_ = policy_output
        return {"logit": policy_logit}

    def compute_critic(self, x):
        values = self.critic(decoder_input=x, only_last=False)
        return {"value": values}

    def compute_actor_critic(self, x):
        policy_output = self.actor(decoder_input=x)
        policy_logit, *_ = policy_output
        values = self.critic(decoder_input=x, only_last=False)
        return {"logit": policy_logit, "value": values}
