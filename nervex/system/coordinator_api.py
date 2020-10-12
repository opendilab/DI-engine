from flask import Flask, request


def create_coordinator_app(coordinator):
    app = Flask(__name__)

    def build_ret(code, info=''):
        return {'code': code, 'info': info}

    @app.route('/coordinator/register_manager', methods=['POST'])
    def register_manager():
        manager_uid = request.json['manager_uid']
        ret_code = coordinator.deal_with_register_manager(manager_uid)
        if ret_code:
            return build_ret(0)
        else:
            return build_ret(1)

    @app.route('/coordinator/register_learner', methods=['POST'])
    def register_learner():
        learner_uid = request.json['learner_uid']
        learner_ip = request.json['learner_ip']
        learner_port = request.json['learner_port']
        world_size = request.json['world_size']
        restore = request.json['restore']
        ret_info = coordinator.deal_with_register_learner(learner_uid, learner_ip, learner_port, world_size, restore)
        if ret_info:
            return build_ret(0, ret_info)
        else:
            return build_ret(1)

    @app.route('/coordinator/ask_for_job', methods=['POST'])
    def get_sample():
        manager_uid = request.json['manager_uid']
        actor_uid = request.json['actor_uid']
        job = coordinator.deal_with_ask_for_job(manager_uid, actor_uid)
        if job:
            return build_ret(0, job)
        else:
            return build_ret(1)

    @app.route('/coordinator/get_metadata', methods=['POST'])
    def get_metadata():
        job_id = request.json['job_id']
        metadata = request.json['metadata']
        ret_code = coordinator.deal_with_get_metadata(job_id, metadata)
        if ret_code:
            return build_ret(0)
        else:
            return build_ret(1)

    @app.route('/coordinator/finish_job', methods=['POST'])
    def finish_job():
        manager_uid = request.json['manager_uid']
        actor_uid = request.json['actor_uid']
        job_id = request.json['job_id']
        result = request.json['result']
        ret_code = coordinator.deal_with_finish_job(manager_uid, actor_uid, job_id, result)
        if ret_code:
            return build_ret(0)
        else:
            return build_ret(1)

    @app.route('/coordinator/ask_for_metadata', methods=['POST'])
    def ask_for_metadata():
        learner_uid = request.json['learner_uid']
        batch_size = request.json['batch_size']
        ret = coordinator.deal_with_ask_for_metadata(learner_uid, batch_size)
        if ret:
            return build_ret(0, ret)
        else:
            return build_ret(1)

    @app.route('/coordinator/send_train_info', methods=['POST'])
    def update_replay_buffer():
        learner_uid = request.json['learner_uid']
        train_info = request.json['train_info']
        ret_code = coordinator.deal_with_train_info(learner_uid, train_info)
        if ret_code:
            return build_ret(0)
        else:
            return build_ret(1)

    @app.route('/coordinator/register_league_manager', methods=['POST'])
    def register_league_manager():
        league_manager_ip = request.json['league_manager_ip']
        player_ids = request.json['player_ids']
        player_ckpts = request.json['player_ckpts']
        ret_code = coordinator.deal_with_register_league_manager(league_manager_ip, player_ids, player_ckpts)
        if ret_code:
            return build_ret(0)
        else:
            return build_ret(1)

    @app.route('/coordinator/ask_learner_to_reset', methods=['POST'])
    def ask_learner_to_reset():
        player_id = request.json['player_id']
        checkpoint_path = request.json['checkpoint_path']
        ret_code = coordinator.deal_with_ask_learner_to_reset(player_id, checkpoint_path)
        if ret_code:
            return build_ret(0)
        else:
            return build_ret(1)

    @app.route('/coordinator/add_launch_info', methods=['POST'])
    def add_launch_info():
        launch_info = request.json['launch_info']
        ret_code = coordinator.deal_with_add_launch_info(launch_info)
        if ret_code:
            return build_ret(0)
        else:
            return build_ret(1)

    @app.route('/coordinator/get_heartbeats', methods=['POST'])
    def get_heartbeats():
        learner_uid = request.json['learner_uid']
        ret_code = coordinator.deal_with_get_heartbeats(learner_uid)
        if ret_code:
            return build_ret(0)
        else:
            return build_ret(1)

    ###################################################################################
    #                                    for debug                                    #
    ###################################################################################

    @app.route('/debug/get_all_manager', methods=['get'])
    def get_all_manager():
        info = coordinator.deal_with_get_all_manager()
        if info:
            return build_ret(0, info)
        else:
            return build_ret(1)

    @app.route('/debug/get_all_learner', methods=['get'])
    def get_all_learner():
        info = coordinator.deal_with_get_all_learner()
        if info:
            return build_ret(0, info)
        else:
            return build_ret(1)

    @app.route('/debug/get_all_job', methods=['get'])
    def get_all_job():
        info = coordinator.deal_with_get_all_job()
        if info:
            return build_ret(0, info)
        else:
            return build_ret(1)

    @app.route('/debug/get_replay_buffer', methods=['post'])
    def get_replay_buffer():
        learner_uid = request.json['learner_uid']
        info = coordinator.deal_with_get_replay_buffer(learner_uid)
        if info:
            return build_ret(0, info)
        else:
            return build_ret(1)

    @app.route('/debug/push_data_to_replay_buffer', methods=['post'])
    def push_data_to_replay_buffer():
        learner_uid = request.json['learner_uid']
        info = coordinator.deal_with_push_data_to_replay_buffer(learner_uid)
        if info:
            return build_ret(0, info)
        else:
            return build_ret(1)

    return app
