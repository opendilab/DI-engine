import os
import sys
import time
import uuid
import numpy as np
from itertools import count
import logging
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

    @app.route('/league/finish_task', methods=['POST'])
    def finish_task():
        task_result = request.json['task_result']
        ret_code = league_manager_wrapper.deal_with_finish_task(task_result)
        if ret_code:
            return build_ret(0)
        else:
            return build_ret(1)

    return app
