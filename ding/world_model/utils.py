def get_rollout_length_scheduler(cfg):
    if cfg.type == 'linear':
        x0 = cfg.rollout_start_step
        x1 = cfg.rollout_end_step
        y0 = cfg.rollout_length_min
        y1 = cfg.rollout_length_max
        w = (y1 - y0) / (x1 - x0)
        b = y0
        return lambda x: int(min(max(w * (x - x0) + b, y0), y1))
    elif cfg.type == 'constant':
        return lambda x: cfg.rollout_length
    else:
        raise NotImplementedError
