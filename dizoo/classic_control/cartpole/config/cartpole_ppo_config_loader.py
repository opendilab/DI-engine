import math

from ding.loader import dict_, is_type, to_type, collection, interval, is_positive, mcmp, enum, item, raw, check_only
from ding.utils import pretty_print

cartpole_ppo_main_loader = dict_(
    env=item('env') >> dict_(
        collector_env_num=item('collector_env_num') >> is_type(int) >> interval(1, 32),
        evaluator_env_num=item('evaluator_env_num') >> is_type(int) >> interval(1, 32),
    ),
    policy=item('policy') >> dict_(
        type=item('type') | raw('dqn') >> is_type(str),
        cuda=item('cuda') >> is_type(bool),
        on_policy=item('on_policy') | raw(False) >> is_type(bool),
        priority=item('priority') | raw(False) >> is_type(bool),
        model=item('model') >> dict_(
            obs_dim=item('obs_shape') >> (is_type(int) | collection(int)),
            action_dim=item('action_shape') >> (is_type(int) | collection(int)),
            embedding_dim=item('embedding_size') >> (is_type(int) | collection(int)),
        ),
        learn=item('learn') >> dict_(
            multi_gpu=item('multi_gpu') | raw(False) >> is_type(bool),
            update_per_collect=item('update_per_collect') | raw(1) >> is_type(int) & interval(1, 500),
            batch_size=item('batch_size') >> (is_type(int) & interval(1, 128)),
            learning_rate=item('learning_rate') >> interval(0.00001, 0.01),
            value_weight=item('value_weight') >> (is_type(float) & interval(0, 1)),
            entropy_weight=item('entropy_weight') >> (is_type(float) & interval(0.00, 0.01)),
            clip_ratio=item('clip_ratio') >> (is_type(float) & interval(0.1, 0.3)),
            adv_norm=item('adv_norm') | raw(False) >> is_type(bool),
        ),
        collect=item('collect') >> dict_(
            n_sample=item('n_sample') >> is_type(int) >> interval(8, 128),
            unroll_len=item('unroll_len') >> is_type(int) >> interval(1, 200),
            discount_factor=item('discount_factor') >> (is_type(float) & interval(0.9, 0.999)),
            gae_lambda=item('gae_lambda') >> (is_type(float) & interval(0.0, 1.0)),
        ),
    ),
)
cartpole_ppo_create_loader = dict_(
    env=item('env') >> dict_(
        import_names=item('import_names') >> collection(str),
        type=item('type') >> is_type(str),
    ),
    env_manager=item('env_manager') >> dict_(
        type=item('type') >> enum('base', 'subprocess', 'async_subprocess'),
        shared_memory=item('shared_memory') | raw(True) >> is_type(bool),
    ),
    policy=item('policy') >> dict_(type=item('type') >> is_type(str), ),
)

if __name__ == "__main__":
    from dizoo.classic_control.cartpole.config import cartpole_ppo_config, cartpole_ppo_create_config
    output = cartpole_ppo_main_loader(cartpole_ppo_config)
    pretty_print(output, direct_print=True)
    output = cartpole_ppo_create_loader(cartpole_ppo_create_config)
    pretty_print(output, direct_print=True)
