from easydict import EasyDict

model_config = dict(
    # ===== Encoder =====
    encoder=dict(
        match_scalar=dict(
            ball_position=dict(input_dim=3, output_dim=32),
            ball_direction=dict(input_dim=3, output_dim=32),
            ball_rotation=dict(input_dim=3, output_dim=32),
            ball_owned_team=dict(input_dim=3, output_dim=32),
            ball_owned_player=dict(input_dim=12, output_dim=32),
            active_player=dict(input_dim=11, output_dim=32),
            designated_player=dict(input_dim=11, output_dim=32),
            active_player_sticky_actions=dict(input_dim=10, output_dim=64),
            score=dict(input_dim=22, output_dim=64),
            steps_left=dict(input_dim=30, output_dim=128),
            game_mode=dict(input_dim=7, output_dim=128),
        ),
        player=dict(
            # choices: ['transformer', 'spatial']
            encoder_type='transformer',
            transformer=dict(
                player_num=22,
                player_attr_dim=dict(
                    team=2, index=11, position=2, direction=2, tired_factor=1, yellow_card=2, active=2, role=10
                ),
                input_dim=1,
                head_dim=64,
                hidden_dim=128,
                output_dim=1,
                head_num=2,
                mlp_num=2,
                layer_num=3,
                dropout_ratio=1
            ),
            spatial=dict(
                resblock_num=4,
                fc_dim=256,
                project_dim=32,
                down_channels=[64, 128],
                activation='relu',
                norm_type='BN',
                scatter_type='add',
                player_attr_dim=dict(
                    team=2, index=11, position=2, direction=2, tired_factor=1, yellow_card=2, active=2, role=10
                ),
            ),
        )
    ),
    # ===== Policy =====
    policy=dict(
        res_block=dict(hidden_dim=1024, block_num=3),
        dqn=dict(dueling=True, a_layer_num=2, v_layer_num=2),
        action_dim=19,
    )
)

default_model_config = EasyDict(model_config)
