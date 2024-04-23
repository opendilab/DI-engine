from dizoo.ising_env.config.ising_mfq_config import main_config, create_config
from ding.entry import eval


def main():
    main_config.env.collector_env_num = 1
    main_config.env.evaluator_env_num = 1
    main_config.env.n_evaluator_episode = 1
    ckpt_path = './ckpt_best.pth.tar'
    replay_path = './replay_videos'
    eval((main_config, create_config), seed=1, load_path=ckpt_path, replay_path=replay_path)


if __name__ == "__main__":
    main()
