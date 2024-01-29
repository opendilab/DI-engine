from dizoo.petting_zoo.config.ptz_simple_spread_mappo_config import main_config, create_config
from ding.entry import eval


def main():
    ckpt_path = './ckpt_best.pth.tar'
    replay_path = './replay_videos'
    eval((main_config, create_config), seed=0, load_path=ckpt_path, replay_path=replay_path)


if __name__ == "__main__":
    main()
