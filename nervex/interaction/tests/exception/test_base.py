import json
from contextlib import contextmanager
from typing import Optional, Mapping, Any

import pytest
import requests
import responses
from requests import HTTPError


class _HTTPErrorGenerator:

    @classmethod
    def _generate_exception(
        cls, code: int, message: str, data: Optional[Mapping[str, Any]] = None, success: bool = False
    ):

        @contextmanager
        def _yield_func():
            with responses.RequestsMock(assert_all_requests_are_fired=False) as rsp:
                rsp.add(
                    **{
                        'method': responses.GET,
                        'url': 'http://example.com/path',
                        'body': json.dumps(
                            {
                                "success": not not success,
                                "code": int(code),
                                "message": str(message),
                                "data": data or {},
                            }
                        ),
                        'status': 400,
                        'content_type': 'application/json',
                    }
                )

                yield

        @responses.activate
        def _get_exception():
            try:
                with _yield_func():
                    response = requests.get('http://example.com/path')
                    response.raise_for_status()
            except HTTPError as err:
                return err
            else:
                pytest.fail('Should not reach here.')

        return _get_exception()
