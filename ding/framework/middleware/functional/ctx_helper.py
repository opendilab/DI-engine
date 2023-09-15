from typing import TYPE_CHECKING, Callable
import os
import pickle
import dataclasses
from ding.framework import task
if TYPE_CHECKING:
    from ding.framework import Context


def final_ctx_saver(name: str) -> Callable:

    def _save(ctx: "Context"):
        if task.finish:
            # make sure the items to be recorded are all kept in the context
            with open(os.path.join(name, 'result.pkl'), 'wb') as f:
                final_data = {
                    'total_step': ctx.total_step,
                    'train_iter': ctx.train_iter,
                    'last_eval_iter': ctx.last_eval_iter,
                    'eval_value': ctx.last_eval_value,
                }
                if ctx.has_attr('env_step'):
                    final_data['env_step'] = ctx.env_step
                    final_data['env_episode'] = ctx.env_episode
                if ctx.has_attr('trained_env_step'):
                    final_data['trained_env_step'] = ctx.trained_env_step
                pickle.dump(final_data, f)

    return _save
