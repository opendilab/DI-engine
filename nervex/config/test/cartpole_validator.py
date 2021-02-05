from nervex.config.loader.dict import dict_
from nervex.config.loader.types import is_type, to_type
from nervex.config.loader.collection import collection
from nervex.config.loader.number import interval, positive
from nervex.config.loader.string import enum
from nervex.config.loader.mapping import item
from nervex.config.loader.utils import keep
from nervex.config.loader.norm import norm
from nervex.config.loader.base import Loader
from nervex.utils import pretty_print


def validate():
    __policy_default_loader = dict_(
        use_cuda=item('use_cuda') >> is_type(bool),
        policy_type=item('policy_type') >> is_type(str),
        import_names=item('import_names') >> collection(str),
        on_policy=item('on_policy') >> is_type(bool),
        model=item('model') >> dict_(
            obs_dim=item('obs_dim') >> (is_type(int) | collection(int)),
            action_dim=item('action_dim') >> (is_type(int) | collection(int))
        ),
        learn=item('learn') >> dict_(
            batch_size=item('batch_size') >> (is_type(int) & interval(1, 128)),
            learning_rate=item('learning_rate') >> interval(0.0001, 0.01),
            weight_decay=item('weight_decay') >> interval(0.0, 0.001),
            algo=item('algo') >> dict_(
                target_update_freq=item('target_update_freq') >> (is_type(int) & interval(100, 2000)),
                discount_factor=item('discount_factor') >> interval(0.9, 1.0),
                nstep=item('nstep') >> (is_type(int) & interval(1, 10)),
            ),
        ),
        collect=item('collect') >> dict_(
            traj_len=item('traj_len') >> (interval(1, 200) | (enum('inf') >> to_type(float))),
            unroll_len=item('unroll_len') >> interval(1, 200),
            algo=item('algo') >> dict_(nstep=item('nstep') >> (is_type(int) & interval(1, 10))),
        ),
        command=item('command') >> dict_(
            eps=item('eps') >> dict_(
                type=item('type') >> enum('linear', 'exp'),
                start=item('start') >> interval(0.0, 1.0, left_ok=False),
                end=item('end') >> interval(0.0, 1.0, right_ok=False),
                decay=item('decay') >> (is_type(int) | (is_type(float) >> to_type(int))) >> positive(),
            )
        ),
    )
    error_loader = Loader((lambda x: x > 0, lambda x: ValueError('value is {x}'.format(x=x))))
    relation_loader = dict_(
        nstep_check=Loader(
            norm(item('learn') >> item('algo') >> item('nstep')) ==
            norm(item('collect') >> item('algo') >> item('nstep'))
        ) >> error_loader,
        unroll_len_check=item('collect') >> (item('unroll_len') <= norm(item('traj_len'))) >> error_loader,
        eps_check=item('command') >> item('eps') >> (norm(item('start')) >= norm(item('end'))) >> error_loader
    ) & keep()
    __policy_default_loader = __policy_default_loader >> relation_loader

    __policy_default_config = dict(
        use_cuda=False,
        policy_type='dqn',
        import_names=['nervex.policy.dqn'],
        on_policy=False,
        model=dict(
            obs_dim=4,
            action_dim=2,
        ),
        learn=dict(
            batch_size=32,
            learning_rate=0.001,
            weight_decay=0.,
            algo=dict(
                target_update_freq=100,
                discount_factor=0.95,
                nstep=1,
            ),
        ),
        collect=dict(
            traj_len=1,
            unroll_len=1,
            algo=dict(nstep=1),
        ),
        command=dict(eps=dict(
            type='exp',
            start=0.95,
            end=0.1,
            decay=1e4,
        ), ),
    )
    output = __policy_default_loader(__policy_default_config)
    pretty_print(output, direct_print=True)


if __name__ == "__main__":
    validate()
