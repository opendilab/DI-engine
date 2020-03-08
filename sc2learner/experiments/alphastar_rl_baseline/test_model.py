from sc2learner.agents.model import build_model
from easydict import EasyDict
import yaml


def main():
    config_path = './config.yaml'
    with open(config_path) as f:
        cfg = yaml.load(f)
    cfg = EasyDict(cfg)
    model = build_model(cfg)
    print(model)


if __name__ == "__main__":
    main()
