from easydict import EasyDict
from nervex.utils import find_free_port


def set_learner_aggregator_config(cfg: EasyDict) -> EasyDict:
    learner_names = []
    for k in cfg.keys():
        if k.startswith('learner'):
            learner_names.append(k[7:])
    aggregator_cfgs = {}
    for n in learner_names:
        if isinstance(list, cfg['learner' + n].port):
            host = 'auto'
            learner_host = cfg['learner' + n].host
            learner_port = cfg['learner' + n].port
            assert len(learner_host) == len(learner_port)
            learner_interaction_cfg = {i: [i, h, p] for i, (h, p) in enumerate(zip(learner_host, learner_port))}
            aggregator_cfg = dict(
                master=dict(
                    host=host,
                    port=find_free_port(),
                ),
                slave=dict(
                    host=host,
                    port=find_free_port(),
                ),
                learner=learner_interaction_cfg,
            )
            aggregator_cfgs[n] = aggregator_cfg

    for n, c in aggregator_cfgs.items():
        cfg['aggregator' + n] = c
    return cfg


def set_learner_interaction_for_coordinator(cfg: EasyDict) -> EasyDict:
    use_aggregator = cfg.get('use_aggregator', False)
    keyword = 'aggregator' if use_aggregator else 'learner'
    cfg.coordinator.interaction.learner = {}
    for k in cfg.keys():
        if k.startswith(keyword):
            n = k[len(keyword):]
            cfg.coordinator.interaction.learner['learner' + n] = ['learner' + n, cfg[k].host, cfg[k].port]
    return cfg
