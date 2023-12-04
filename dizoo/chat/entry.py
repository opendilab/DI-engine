from easydict import EasyDict

from ding.bonus.ppof import PPOF
from ding.model.template import LlamaVAC


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--actor_path', type=str)
    parser.add_argument('--critic_path', type=str)
    parser.add_argument('--tokenizer_path', type=str)
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
        tokenizer_path=args.tokenizer_path,
        opt=opt
    )

    policy = PPOF(
        env_id="chat",
        exp_name="rlhf-ppo",
        model=model
    )
    policy.train(collector_env_num=1, evaluator_env_num=1, debug=True)
