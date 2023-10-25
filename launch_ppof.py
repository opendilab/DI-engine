from easydict import EasyDict
from transformers import LlamaTokenizer

from ding.bonus.ppof import PPOF
from ding.model.template.vac import LlamaVAC


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


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--actor_path', type=str)
    parser.add_argument('--critic_path', type=str)
    args = parser.parse_args()

    opt = EasyDict({
        "maxlen_res": 512,
        "temperature": 1,
        "repetition_penalty": 1,
        "topp": 0.8
    })

    model = LlamaVAC(
        actor_path=args.actor_path,
        critic_path=args.critic_path,
        tokenizer=get_tokenizer(args.actor_path),
        opt=opt
    )

    policy = PPOF(
        env_id="prompt-generator",
        exp_name="rlhf-ppo",
        model=model
    )
    policy.train()
