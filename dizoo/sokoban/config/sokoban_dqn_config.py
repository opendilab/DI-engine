from easydict import EasyDict

sokoban_dqn_config = dict(
    exp_name = "sokoban_dqn_seed0",
    env = dict(
        collctor_env_num=8,
        evaluator_env_num=8,
        n_evaluator_episode=8,
        env_id="sokoban",
        dim_room = (10,10),
        max_steps=120,
        num_boxes=4,
        num_gen_steps=None,
        reset=True,
    )
    policy = dict(
        cuda = True,
        model = dict(

        ),
        learn = dict(

        ),
        collect = dict(
            penalty_for_step = -0.1,
            penalty_box_off_target = -1,
            reward_box_on_target = 1,
            reward_finished = 10,
            reward_last = 0,
        ),
        other = dict(
            viewer = None,
            max_steps = max_steps,
            render_mode = 'rgb_array',
            observation_mode='rgb_array',
        )
    )
)