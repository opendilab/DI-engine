import json
import time
from contextlib import contextmanager
from multiprocessing import Process

import pytest
import requests
import responses
from flask import Flask, request
from requests import HTTPError
from urlobject import URLObject

from ..test_utils import silence
from ...base import get_host_ip, success_response, get_values_from_response, split_http_address, HttpEngine, \
    get_http_engine_class

app = Flask('_test_get_host_ip')


@app.route('/ping', methods=['GET'])
def ping_method():
    return success_response(message='PONG!')


@app.route('/shutdown', methods=['DELETE'])
def shutdown_method():
    _shutdown_func = request.environ.get('werkzeug.server.shutdown')
    if _shutdown_func is None:
        raise RuntimeError('Not running with the Werkzeug Server')

    _shutdown_func()
    return success_response(message='Shutdown request received, this server will be down later.')


_APP_PORT = 17503


def run_test_app():
    with silence():
        app.run(host='0.0.0.0', port=_APP_PORT)


@pytest.mark.unittest
class TestInteractionBaseNetwork:

    @pytest.mark.execution_timeout(5.0, method='thread')
    def test_get_host_ip(self):
        app_process = Process(target=run_test_app)
        app_process.start()

        _local_ip = get_host_ip()
        _local_server_host = URLObject().with_scheme('http').with_hostname(_local_ip).with_port(_APP_PORT)

        try:
            _start_time = time.time()
            _start_complete = False
            while not _start_complete and time.time() - _start_time < 5.0:
                try:
                    response = requests.get(_local_server_host.add_path('/ping'))
                    if response.ok:
                        _start_complete = True
                        break
                    time.sleep(0.2)
                except (requests.exceptions.BaseHTTPError, requests.exceptions.RequestException):
                    time.sleep(0.2)

            if not _start_complete:
                pytest.fail('Test server start failed.')

            assert get_values_from_response(response) == (
                200,
                True,
                0,
                'PONG!',
                None,
            )
        finally:
            try:
                requests.delete(_local_server_host.add_path('/shutdown'))
            finally:
                app_process.join()

    def test_split_http_address(self):
        assert split_http_address('http://1.2.3.4') == ('1.2.3.4', 80, False, '')
        assert split_http_address('https://1.2.3.4') == ('1.2.3.4', 443, True, '')
        assert split_http_address('http://1.2.3.4:8888') == ('1.2.3.4', 8888, False, '')
        assert split_http_address('https://1.2.3.4:8787/this/is/path') == ('1.2.3.4', 8787, True, '/this/is/path')


@pytest.mark.unittest
class TestInteractionBaseHttpEngine:

    @contextmanager
    def __yield_http_engine(self):
        with responses.RequestsMock(assert_all_requests_are_fired=False) as rsp:
            rsp.add(
                **{
                    'method': responses.GET,
                    'url': 'http://example.com:7777/this/is/404',
                    'body': json.dumps({"exception": "reason"}),
                    'status': 404,
                    'content_type': 'application/json',
                }
            )
            rsp.add(
                **{
                    'method': responses.GET,
                    'url': 'http://example.com:7777/this/is/200',
                    'body': json.dumps({"success": True}),
                    'status': 200,
                    'content_type': 'application/json',
                }
            )

            yield

    @responses.activate
    def test_http_engine_basic(self):
        with self.__yield_http_engine():
            engine = HttpEngine(host='example.com', port=7777)
            response = engine.request('GET', '/this/is/200')
            assert response.status_code == 200
            assert json.loads(response.content.decode()) == {"success": True}

            with pytest.raises(HTTPError) as ei:
                engine.request('GET', '/this/is/404')

            err = ei.value
            assert err.response.status_code == 404
            assert json.loads(err.response.content.decode()) == {'exception': 'reason'}

    @responses.activate
    def test_http_engine_with_path(self):
        with self.__yield_http_engine():
            engine = HttpEngine(host='example.com', port=7777, path='/this/is')
            response = engine.request('GET', '200')
            assert response.status_code == 200
            assert json.loads(response.content.decode()) == {"success": True}

            with pytest.raises(HTTPError) as ei:
                engine.request('GET', '404')

            err = ei.value
            assert err.response.status_code == 404
            assert json.loads(err.response.content.decode()) == {'exception': 'reason'}

    @responses.activate
    def test_get_http_engine_class(self):
        with self.__yield_http_engine():
            _token = '233'

            _http_engine_class = get_http_engine_class(
                headers={'Token': lambda: _token},
                data_processor=(lambda d: {
                    'data': json.dumps(d)
                }),
                http_error_gene=lambda e: RuntimeError('This is {status}'.format(status=e.response.status_code))
            )()
            engine = _http_engine_class(host='example.com', port=7777, path='/this/is')

            response = engine.request('GET', '200', {'a': 'skdjgflksdj'})
            assert response.status_code == 200
            assert json.loads(response.content.decode()) == {"success": True}
            assert response.request.headers['Token'] == '233'
            assert json.loads(response.request.body) == {'data': json.dumps({'a': 'skdjgflksdj'})}

            with pytest.raises(RuntimeError) as ei:
                engine.request('GET', '404', {'a': 'skdjgflksdj'})

            err = ei.value
            assert 'This is 404' in str(err)
