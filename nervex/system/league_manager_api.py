from flask import Flask, request


def create_league_manager_app(league_manager_wrapper):
    app = Flask(__name__)

    def build_ret(code, info=''):
        return {'code': code, 'info': info}

    @app.route('/league/run_league', methods=['GET'])
    def run_league():
        ret_code = league_manager_wrapper.deal_with_run_league()
        if ret_code:
            return build_ret(0)
        else:
            return build_ret(1)

    @app.route('/league/finish_job', methods=['POST'])
    def finish_job():
        job_result = request.json['job_result']
        ret_code = league_manager_wrapper.deal_with_finish_job(job_result)
        if ret_code:
            return build_ret(0)
        else:
            return build_ret(1)

    @app.route('/league/update_active_player', methods=['POST'])
    def update_active_player():
        player_info = request.json['player_info']
        ret_code = league_manager_wrapper.deal_with_update_active_player(player_info)
        if ret_code:
            return build_ret(0)
        else:
            return build_ret(1)

    return app
