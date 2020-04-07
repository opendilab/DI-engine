import os
import sys
import time
import uuid

import numpy as np
from itertools import count
import logging
from flask import Flask, request

from sc2learner.utils.log_helper import TextLogger


def create_manager_app(manager):
    app = Flask(__name__)


    def build_ret(code, info=''):
        return {'code': code, 'info': info}


    @app.route('/manager/register', methods=['POST'])
    def register():
        actor_uid = request.json['actor_uid']
        ret_code = manager.deal_with_register_actor(actor_uid)
        if ret_code:
            return build_ret(0)
        else:
            return build_ret(1)


    @app.route('/manager/ask_for_job', methods=['POST'])
    def get_sample():
        actor_uid = request.json['actor_uid']
        job = manager.deal_with_ask_for_job(actor_uid)
        if job:
            return build_ret(0, job)
        else:
            return build_ret(1)


    @app.route('/manager/get_metadata', methods=['POST'])
    def finish_sample():
        actor_uid = request.json['actor_uid']
        job_id = request.json['job_id']
        metadata = request.json['metadata']
        ret_code = manager.deal_with_get_metadata(actor_uid, job_id, metadata)
        if ret_code:
            return build_ret(0)
        else:
            return build_ret(1)


@app.route('/manager/finish_job', methods=['POST'])
def finish_job():
    actor_uid = request.json['actor_uid']
    job_id = request.json['job_id']
    ret_code = manager.deal_with_finish_job(actor_uid, job_id)
    if ret_code:
        return build_ret(0)
    else:
        return build_ret(1)


    @app.route('/manager/get_heartbeats', methods=['POST'])
    def get_heartbeats():
        actor_uid = request.json['actor_uid']
        ret_code = manager.deal_with_get_heartbeats(actor_uid)
        if ret_code:
            return build_ret(0)
        else:
            return build_ret(1)


    ###################################################################################
    #                                                                                 #
    #                                    for debug                                    #
    #                                                                                 #
    ###################################################################################


    @app.route('/debug/get_all_actor', methods=['get'])
    def get_all_manager():
        info = manager.deal_with_get_all_actor_from_debug()
        if info:
            return build_ret(0, info)
        else:
            return build_ret(1)


    @app.route('/debug/get_all_job', methods=['get'])
    def get_all_job():
        info = manager.deal_with_get_all_job_from_debug()
        if info:
            return build_ret(0, info)
        else:
            return build_ret(1)


    @app.route('/debug/get_reuse_job_list', methods=['get'])
    def get_reuse_job_list():
        info = manager.deal_with_get_reuse_job_list_from_debug()
        if info:
            return build_ret(0, info)
        else:
            return build_ret(1)


    @app.route('/debug/clear_reuse_job_list', methods=['get'])
    def clear_reuse_job_list():
        info = manager.deal_with_clear_reuse_job_list_from_debug()
        if info:
            return build_ret(0, info)
        else:
            return build_ret(1)


    @app.route('/debug/launch_actor', methods=['POST'])
    def launch_actor():
        use_partitions = request.json['use_partitions']
        actor_num = request.json['actor_num']
        info = manager.deal_with_launch_actor()
        if info:
            return build_ret(0, info)
        else:
            return build_ret(1)
    return app
