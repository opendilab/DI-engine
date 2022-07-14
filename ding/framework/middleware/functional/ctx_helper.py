from typing import TYPE_CHECKING, Callable
import os
import pickle
from ding.framework import task
if TYPE_CHECKING:
    from ding.framework import Context


def final_ctx_saver(name: str) -> Callable:

    def _save(ctx: "Context"):
        if task.finish:
            with open(os.path.join(name, 'result.pkl'), 'wb') as f:
                final_data = {
                    'total_step': ctx.total_step,
                    'train_iter': ctx.train_iter,
                    'eval_value': ctx.eval_value,
                }
                if 'env_step' in ctx:
                    final_data['env_step'] = ctx.env_step
                    final_data['env_episode'] = ctx.env_episode
                pickle.dump(final_data, f)

    return _save
