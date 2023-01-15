import os

from ditk import logging

from lighttuner.hpo import R, uniform, choice
from lighttuner.hpo import hpo
from lighttuner.scheduler import run_scheduler_local


def demo():
    dir_name = os.path.abspath('./benchmark')

    with run_scheduler_local(task_config_template_path=os.path.join(dir_name, "hopper_td3_wandb_pipeline.py"),
                             dijob_project_name="hopper_td3_wandb",max_number_of_running_task=4) as scheduler:

        opt = hpo(scheduler.get_hpo_callable())
        cfg, ret, metrics = opt.grid() \
            .max_steps(5) \
            .max_workers(4) \
            .maximize(R['eval_value']) \
            .spaces({'seed': choice([0,1])}).run()
        print(cfg)
        print(ret)


if __name__ == "__main__":
    logging.try_init_root(logging.INFO)
    demo()
