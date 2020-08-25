import os
import sys
import time
import uuid
import numpy as np
from itertools import count
import logging
from flask import Flask, request


def create_league_manager_app(league_manager):
    app = Flask(__name__)

    def build_ret(code, info=''):
        return {'code': code, 'info': info}

    @app.route('/league/run_league', methods=['GET'])
    def run_league():
        ret_code = league_manager.deal_with_run_league()
        if ret_code:
            return build_ret(0)
        else:
            return build_ret(1)

    @app.route('/league/finish_match', methods=['POST'])
    def finish_match():
        match_info = request.json['match_info']
        ret_code = league_manager.deal_with_finish_match(match_info)
        if ret_code:
            return build_ret(0)
        else:
            return build_ret(1)

    return app
