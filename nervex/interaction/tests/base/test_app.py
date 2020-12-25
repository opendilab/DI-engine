import json

import pytest
from flask import Flask

from ...base import success_response, failure_response, get_values_from_response, ResponsibleException, responsible


@pytest.mark.unittest
class TestInteractionBaseApp:

    def test_success_response(self):
        app = Flask('_test_success_response')

        @app.route('/success', methods=['GET'])
        def success_method():
            return success_response(
                data={
                    'a': 1,
                    'b': 2,
                    'sum': 3,
                },
                message='This is success message.',
            )

        client = app.test_client()

        response = client.get('/success')
        assert response.status_code == 200
        assert json.loads(response.data.decode()) == {
            'success': True,
            'code': 0,
            'data': {
                'a': 1,
                'b': 2,
                'sum': 3,
            },
            'message': 'This is success message.',
        }

    # noinspection DuplicatedCode
    def test_failure_response(self):
        app = Flask('_test_failure_response')

        @app.route('/fail', methods=['GET'])
        def fail_method():
            return failure_response(
                code=233,
                message='This is failure message.',
                data={
                    'a': 2,
                    'b': 3,
                    'sum': 5,
                },
            ), 404

        client = app.test_client()

        response = client.get('/fail')
        assert response.status_code == 404
        assert json.loads(response.data.decode()) == {
            'success': False,
            'code': 233,
            'data': {
                'a': 2,
                'b': 3,
                'sum': 5,
            },
            'message': 'This is failure message.',
        }

    def test_get_values_from_response(self):
        app = Flask('_test_get_values_from_response')

        @app.route('/success', methods=['GET'])
        def success_method():
            return success_response(
                data={
                    'a': 1,
                    'b': 2,
                    'sum': 3,
                },
                message='This is success message.',
            )

        @app.route('/fail', methods=['GET'])
        def fail_method():
            return failure_response(
                code=233,
                message='This is failure message.',
                data={
                    'a': 2,
                    'b': 3,
                    'sum': 5,
                },
            ), 404

        client = app.test_client()

        response = client.get('/success')
        assert response.status_code == 200
        assert get_values_from_response(response) == (
            200,
            True,
            0,
            'This is success message.',
            {
                'a': 1,
                'b': 2,
                'sum': 3,
            },
        )

        response = client.get('/fail')
        assert response.status_code == 404
        assert get_values_from_response(response) == (
            404,
            False,
            233,
            'This is failure message.',
            {
                'a': 2,
                'b': 3,
                'sum': 5,
            },
        )


@pytest.mark.unittest
class TestInteractionBaseResponsibleException:
    # noinspection DuplicatedCode
    def test_it(self):

        class NotFound(ResponsibleException):

            def __init__(self):
                ResponsibleException.__init__(
                    self=self,
                    status_code=404,
                    code=233,
                    message='This is failure message.',
                    data={
                        'a': 2,
                        'b': 3,
                        'sum': 5,
                    }
                )

        class AccessDenied(ResponsibleException):

            def __init__(self):
                ResponsibleException.__init__(
                    self=self,
                    status_code=403,
                    code=322,
                    message='This is another failure message.',
                    data={
                        'a': 2,
                        'b': 3,
                        'sum': 7,
                    }
                )

        app = Flask('_test_failure_response')

        @app.route('/fail', methods=['GET'])
        @responsible(classes=(NotFound, ))
        def fail_method():
            raise NotFound

        @app.route('/403', methods=['GET'])
        @responsible()
        def denied_method():
            raise AccessDenied

        @app.route('/success', methods=['GET'])
        @responsible()
        def success_method():
            return success_response(
                data={
                    'a': 1,
                    'b': 2,
                    'sum': 3,
                },
                message='This is success message.',
            )

        client = app.test_client()

        response = client.get('/fail')
        assert response.status_code == 404
        assert json.loads(response.data.decode()) == {
            'success': False,
            'code': 233,
            'data': {
                'a': 2,
                'b': 3,
                'sum': 5,
            },
            'message': 'This is failure message.',
        }

        response = client.get('/403')
        assert response.status_code == 403
        assert json.loads(response.data.decode()) == {
            'success': False,
            'code': 322,
            'data': {
                'a': 2,
                'b': 3,
                'sum': 7,
            },
            'message': 'This is another failure message.',
        }

        response = client.get('/success')
        assert response.status_code == 200
        assert json.loads(response.data.decode()) == {
            'success': True,
            'code': 0,
            'data': {
                'a': 1,
                'b': 2,
                'sum': 3,
            },
            'message': 'This is success message.',
        }
