import json
import socket
from typing import Optional, Any, Mapping, Callable, Type, Tuple

import requests
from requests import HTTPError
from urlobject import URLObject
from urlobject.path import URLPath

from .common import translate_dict_func


def get_host_ip() -> Optional[str]:
    s = None
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    finally:
        if s is not None:
            s.close()
    return ip


_DEFAULT_HTTP_PORT = 80
_DEFAULT_HTTPS_PORT = 443


def split_http_address(address: str, default_port: Optional[int] = None) -> Tuple[str, int, bool, str]:
    _url = URLObject(address)

    _host = _url.hostname
    _https = (_url.scheme.lower()) == 'https'
    _port = _url.port or default_port or (_DEFAULT_HTTPS_PORT if _https else _DEFAULT_HTTP_PORT)
    _path = str(_url.path) or ''

    return _host, _port, _https, _path


class HttpEngine:

    def __init__(self, host: str, port: int, https: bool = False, path: str = None):
        self.__base_url = URLObject().with_scheme('https' if https else 'http') \
            .with_hostname(host).with_port(port).add_path(path or '')
        self.__session = requests.session()

    # noinspection PyMethodMayBeStatic
    def _data_process(self, data: Optional[Mapping[str, Any]] = None) -> Mapping[str, Any]:
        return data or {}

    # noinspection PyMethodMayBeStatic
    def _base_headers(self) -> Mapping[str, None]:
        return {}

    def _error_handler(self, err: Exception):
        raise err

    def get_url(self, path: str = None):
        original_segments = self.__base_url.path.segments
        path_segments = URLPath().add(path or '').segments
        return str(self.__base_url.with_path(URLPath.join_segments(original_segments + path_segments)))

    def request(
            self,
            method: str,
            path: str,
            data: Optional[Mapping[str, Any]] = None,
            headers: Optional[Mapping[str, Any]] = None,
            raise_for_status: bool = True
    ) -> requests.Response:
        _headers = dict(self._base_headers())
        _headers.update(headers or {})

        try:
            response = self.__session.request(
                method=method,
                url=self.get_url(path),
                data=json.dumps(self._data_process(data) or {}),
                headers=_headers or {},
            )
            if raise_for_status:
                response.raise_for_status()
        except Exception as e:
            self._error_handler(e)
        else:
            return response


def get_http_engine_class(
        headers: Mapping[str, Callable[..., Any]],
        data_processor: Optional[Callable[[Mapping[str, Any]], Mapping[str, Any]]] = None,
        http_error_gene: Optional[Callable[[
            HTTPError,
        ], None]] = None,
) -> Callable[..., Type[HttpEngine]]:

    def _func(*args, **kwargs) -> Type[HttpEngine]:

        class _HttpEngine(HttpEngine):

            def _data_process(self, data: Optional[Mapping[str, Any]] = None) -> Mapping[str, Any]:
                return (data_processor or (lambda d: d or {}))(data or {})

            def _base_headers(self) -> Mapping[str, None]:
                return translate_dict_func(headers)(*args, **kwargs)

            def _error_handler(self, err: Exception):
                if http_error_gene is not None and isinstance(err, HTTPError):
                    raise http_error_gene
                else:
                    raise err

        return _HttpEngine

    return _func
