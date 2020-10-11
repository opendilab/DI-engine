import argparse
import multiprocessing
from multiprocessing import Process

from nervex.utils import read_config
from nervex.worker.actor import create_actor

multiprocessing.set_start_method('spawn', force=True)


def run_actor(cfg):
    actor = create_actor(cfg)
    actor.run()
    actor.close()


def main(cfg):
    ps = [Process(target=run_actor, args=(cfg,)) for _ in range(cfg.system.actor_num)]
    for p in ps:
        p.start()
    for p in ps:
        p.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="nervex actor start point")
    parser.add_argument("--config", dest="config", required=True, help="actor settings in yaml format")
    args = parser.parse_args()
    cfg = read_config(args.config)
    main(cfg)
