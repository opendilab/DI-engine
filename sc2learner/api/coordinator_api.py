import os
import sys
import time
import uuid

import numpy as np
from itertools import count
import logging
from flask import Flask, request


def create_coordinator_app(coordinator):
    app = Flask(__name__)

    def build_ret(code, info=''):
        return {'code': code, 'info': info}

    @app.route('/coordinator/register_model', methods=['POST'])
    def register_model():
        learner_uid = request.json['learner_uid']
        model_name = request.json['model_name']
        ret_code = coordinator.deal_with_register_model(learner_uid, model_name)
        if ret_code:
            return build_ret(0)
        else:
            return build_ret(1)

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
        ret_code = coordinator.deal_with_register_learner(learner_uid, learner_ip)
        if ret_code:
            return build_ret(0)
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
        manager_uid = request.json['manager_uid']
        actor_uid = request.json['actor_uid']
        job_id = request.json['job_id']
        metadata = request.json['metadata']
        ret_code = coordinator.deal_with_get_metadata(manager_uid, actor_uid, job_id, metadata)
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

    @app.route('/coordinator/update_replay_buffer', methods=['POST'])
    def update_replay_buffer():
        update_info = request.json['update_info']
        ret_code = coordinator.deal_with_update_replay_buffer(update_info)
        if ret_code:
            return build_ret(0)
        else:
            return build_ret(1)

    @app.route('/coordinator/get_learner_train_step', methods=['POST'])
    def get_learner_train_step():
        learner_uid = request.json['learner_uid']
        train_step = request.json['train_step']
        ret_code = coordinator.deal_with_get_learner_train_step(learner_uid, train_step)
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

    return app
