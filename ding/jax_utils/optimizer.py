from typing import Dict, Tuple
import optax


class AdamW:

    def __init__(self, param, lr: float, wd: float, b1: float = 0.9, b2: float = 0.999, eps: float = 1e-08) -> None:
        self.core = optax.adamw(lr, b1=b1, b2=b2, eps=eps, weight_decay=wd)
        self.opt_state = self.core.init(param)
        self.step_count = 0

    def step(self, grad, param, log_optim: bool = False) -> Tuple[Dict, Dict]:
        update, self.opt_state = self.core.update(grad, self.opt_state, param)
        param = optax.apply_updates(param, update)
        self.step_count += 1

        info = {}
        if log_optim:
            info['grad_norm'] = optax.global_norm(grad)
            info['update_norm'] = optax.global_norm(update)
        return param, info

    def state_dict(self) -> Dict:
        return {'opt_state': self.opt_state, 'step': self.step_count}

    def load_state_dict(self, state_dict: Dict) -> None:
        self.opt_state = state_dict['opt_state']
        self.step_count = state_dict['step']


def periodic_update(param, target_param, step: int, freq: int):
    if step % freq == 0:
        return param
    else:
        return target_param
