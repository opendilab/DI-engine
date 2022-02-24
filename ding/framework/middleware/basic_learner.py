from typing import TYPE_CHECKING
from ding.policy import Policy
from ding.worker.buffer import Buffer
import logging

if TYPE_CHECKING:
    from ding.framework import Task, Context


def basic_learner(task: "Task", cfg: dict, policy: Policy, buffer_: Buffer):

    def _learn(ctx: "Context"):
        ctx.setdefault("train_iter", 0)
        ctx.keep("train_iter")

        for _ in range(cfg.policy.learn.update_per_collect):
            try:
                buffered_data = buffer_.sample(policy.learn_mode.get_attribute('batch_size'))
            except ValueError:
                break
            data = [d.data for d in buffered_data]
            learn_output = policy.learn_mode.forward(data)
            if ctx.train_iter % 20 == 0:
                logging.info(
                    'Current Training: Train Iter({})\tLoss({:.3f})'.format(ctx.train_iter, learn_output['total_loss'])
                )
            ctx.train_iter += 1

    return _learn
