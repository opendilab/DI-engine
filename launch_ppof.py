from ding.bonus.ppof import PPOF
from ding.model.template.vac import LlamaVAC

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--actor_path', type=str)
    parser.add_argument('--critic_path', type=str)
    args = parser.parse_args()
    model = LlamaVAC(
        actor_path=args.actor_path,
        critic_path=args.critic_path
    )

    policy = PPOF(
        env_id="prompt-generator",
        exp_name="rlhf-ppo",
        model=model
    )
    policy.train()
