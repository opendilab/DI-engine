from typing import Union, Optional, List, Any, Tuple
import torch

from ding.entry.utils import random_collect
from ding.world_model import mbrl_entry_setup


def serial_pipeline_dream(
        input_cfg: Union[str, Tuple[dict, dict]],
        seed: int = 0,
        env_setting: Optional[List[Any]] = None,
        model: Optional[torch.nn.Module] = None,
        max_train_iter: Optional[int] = int(1e10),
        max_env_step: Optional[int] = int(1e10),
):
    cfg, policy, world_model, env_buffer, learner, collector, collector_env, evaluator, commander, tb_logger = \
        mbrl_entry_setup(input_cfg, seed, env_setting, model, max_train_iter, max_env_step)

    if cfg.policy.get('random_collect_size', 0) > 0:
        random_collect(cfg.policy, policy, collector, collector_env, commander, env_buffer)


    while True:
        # eval the policy
        if evaluator.should_eval(collector.envstep):
            stop, reward = evaluator.eval(learner.save_checkpoint, learner.train_iter, collector.envstep)
            if stop:
                break

        # fill environment buffer
        data = collector.collect(train_iter=learner.train_iter)
        env_buffer.push(data, cur_collector_envstep=collector.envstep)

        # eval&train world model and fill imagination buffer  
        if world_model.should_eval(collector.envstep):
            world_model.eval(env_buffer, collector.envstep, learner.train_iter)
        if world_model.should_train(collector.envstep):
            world_model.train(env_buffer, collector.envstep, learner.train_iter)
        
        update_per_collect = cfg.policy.learn.update_per_collect // world_model.rollout_length_scheduler(collector.envstep)
        update_per_collect = max(1, update_per_collect)
        for i in range(update_per_collect):
            batch_size = learner.policy.get_attribute('batch_size')
            train_data = env_buffer.sample(batch_size, learner.train_iter)
            # dreamer-style: use pure on-policy imagined rollout to train policy, 
            # which depends on the current envstep to decide the rollout length
            learner.train(
                train_data, collector.envstep, 
                policy_kwargs=dict(
                    world_model=world_model,
                    envstep=collector.envstep
                )
            )

        if collector.envstep >= max_env_step or learner.train_iter >= max_train_iter:
            break
    