import os
import sys
import time
import uuid

import numpy as np
from itertools import count
import logging
import argparse
import yaml
from flask import Flask, request

from manager import Manager
from utils.log_helper import TextLogger

parser = argparse.ArgumentParser(description='implementation of AlphaStar')
parser.add_argument('--config', dest='config', required=True, help='settings of AlphaStar in yaml format')
args = parser.parse_args()

# experiments/xxx/api-log/learner-api.log
log_path = os.path.dirname(args.config)
api_dir_name = 'api-log'
log_path = os.path.join(log_path, api_dir_name)
logger = TextLogger(log_path, name="coordinator.log")

cfg = yaml.load(open(args.config, 'r'))
api_info = cfg['api']
learner_port = api_info['learner_port']
coordinator_ip = api_info['coordinator_ip']
coordinator_port = api_info['coordinator_port']
manager_ip = api_info['manager_ip']
manager_port = api_info['manager_port']

manager = Manager(cfg)

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


if __name__ == '__main__':
    app.run(host=manager_ip, port=manager_port, debug=True, use_reloader=False)
