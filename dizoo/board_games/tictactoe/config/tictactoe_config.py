from easydict import EasyDict
from ding.rl_utils.mcts.game_base_config import GameBaseConfig, DiscreteSupport

game_config = EasyDict(dict(
    env_name='tictactoe',
    model_type='board_game',
    device='cuda',
    # debug
    # device='cpu',
    mcts_ctree=False,
    # TODO: mcts_ctree now only support env_num=1, because in cpp MCTS root node,
    #  we must specify the one same action mask,
    #  when env_num>1, the action mask for different env may be different.
    battle_mode='two_player_mode',
    game_history_length=9,
    # battle_mode='one_player_mode',
    # game_history_length=5,
    image_based=False,
    cvt_string=False,
    clip_reward=True,
    game_wrapper=True,
    action_space_size=int(3 * 3),
    amp_type='none',
    obs_shape=(12, 3, 3),  # if frame_stack_num=4
    frame_stack_num=4,
    # obs_shape=(3, 3, 3),  # if frame_stack_num=1
    # frame_stack_num=1,

    image_channel=3,
    gray_scale=False,
    downsample=False,
    vis_result=True,
    # TODO(pu): test the effect of augmentation
    use_augmentation=True,
    # use_augmentation=False,
    # Style of augmentation
    # choices=['none', 'rrc', 'affine', 'crop', 'blur', 'shift', 'intensity']
    augmentation=['shift', 'intensity'],

    # debug
    # collector_env_num=2,
    # evaluator_env_num=2,
    # num_simulations=5,
    # batch_size=5,
    # total_transitions=int(3e3),
    # lstm_hidden_size=64,
    # td_steps=2,
    # num_unroll_steps=3,
    # lstm_horizon_len=3,

    collector_env_num=8,
    evaluator_env_num=5,
    num_simulations=25,
    # batch_size=256,
    batch_size=64,
    # total_transitions=int(1e5),
    total_transitions=int(3e3),
    lstm_hidden_size=64,
    # td_steps=2,
    # to make sure the value target is the final outcome
    td_steps=9,
    num_unroll_steps=3,
    lstm_horizon_len=3,

    # TODO(pu): why 0.99?
    revisit_policy_search_rate=0.99,

    # TODO(pu): why not use adam?
    # lr_manually=True,
    lr_manually=False,

    # TODO(pu): if true, no priority to sample
    use_max_priority=True,  # if true, sample without priority
    # use_max_priority=False,
    use_priority=False,

    # TODO(pu): only used for adjust temperature manually
    max_training_steps=int(1e5),
    auto_temperature=False,
    # only effective when auto_temperature=False
    fixed_temperature_value=0.25,
    # TODO(pu): whether to use root value in reanalyzing?
    use_root_value=False,

    # TODO(pu): test the effect
    last_linear_layer_init_zero=True,
    state_norm=False,

    mini_infer_size=2,
    # (Float type) How much prioritization is used: 0 means no prioritization while 1 means full prioritization
    priority_prob_alpha=0.6,
    # (Float type)  How much correction is used: 0 means no correction while 1 means full correction
    # TODO(pu): test effect of 0.4->1
    priority_prob_beta=0.4,
    prioritized_replay_eps=1e-6,

    root_dirichlet_alpha=0.3,
    root_exploration_fraction=0.25,
    auto_td_steps=int(0.3 * 2e5),
    auto_td_steps_ratio=0.3,

    # UCB formula
    pb_c_base=19652,
    pb_c_init=1.25,

    support_size=10,
    value_support=DiscreteSupport(-10, 10, delta=1),
    reward_support=DiscreteSupport(-10, 10, delta=1),
    max_grad_norm=10,

    test_interval=10000,
    log_interval=1000,
    vis_interval=1000,
    checkpoint_interval=100,
    target_model_interval=200,
    save_ckpt_interval=10000,
    discount=1,
    dirichlet_alpha=0.3,
    value_delta_max=0.01,
    num_actors=1,
    # network initialization/ & normalization
    episode_life=True,
    start_transitions=8,
    transition_num=1,
    # frame skip & stack observation
    frame_skip=4,
    # coefficient
    # TODO(pu): test the effect of value_prefix_loss and consistency_loss
    # reward_loss_coeff=1,  # value_prefix_loss
    reward_loss_coeff=0,  # value_prefix_loss
    value_loss_coeff=0.25,
    policy_loss_coeff=1,
    # consistency_coeff=2,
    consistency_coeff=0,

    bn_mt=0.1,
    # siamese
    # proj_hid=128,
    # proj_out=128,
    # pred_hid=64,
    # pred_out=128,
    proj_hid=32,
    proj_out=32,
    pred_hid=16,
    pred_out=32,

    blocks=1,  # Number of blocks in the ResNet
    channels=16,  # Number of channels in the ResNet
    reduced_channels_reward=16,  # x36 Number of channels in reward head
    reduced_channels_value=16,  # x36 Number of channels in value head
    reduced_channels_policy=16,  # x36 Number of channels in policy head
    resnet_fc_reward_layers=[8],  # Define the hidden layers in the reward head of the dynamic network
    resnet_fc_value_layers=[8],  # Define the hidden layers in the value head of the prediction network
    resnet_fc_policy_layers=[8],  # Define the hidden layers in the policy head of the prediction network
))

game_config = GameBaseConfig(game_config)
