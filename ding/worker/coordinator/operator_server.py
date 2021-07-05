from typing import Optional, Mapping, Any
from requests.exceptions import RequestException
from ding.interaction.base import get_http_engine_class, get_values_from_response


class OperatorServer:

    def __init__(
        self,
        host: str,
        port: Optional[int] = None,
        api_version: str = "v1alpha1",
        https: bool = False,
        namespace: str = None,
        name: str = None,
    ):
        # request part
        self.__http_engine = get_http_engine_class(headers={})()(host, port, https)
        self.__api_version = api_version
        self.__namespace = namespace
        self.__my_name = name
        self.__worker_type = None

    @property
    def api_version(self):
        return self.__api_version

    def set_worker_type(self, type):
        assert type in ['coordinator', 'aggregator'], "invalid worker_type: {}".format(type)
        self.__worker_type = type

    def __prefix_with_api_version(self, path):
        return self.__api_version + path

    def get_replicas(self, name: str = None):
        try:
            if name is None:
                assert self.__worker_type, "set worker type first"
                params = {"namespace": self.__namespace, self.__worker_type: self.__my_name}
            else:
                params = {"namespace": self.__namespace, "name": name}
            response = self.__http_engine.request('GET', self.__prefix_with_api_version('/replicas'), params=params)
        except RequestException as err:
            return self._error_request(err)
        else:
            return self._after_request(*get_values_from_response(response))

    def post_replicas(self, data):
        try:
            data.update({"namespace": self.__namespace, "coordinator": self.__my_name})
            response = self.__http_engine.request('POST', self.__prefix_with_api_version('/replicas'), data=data)
        except RequestException as err:
            return self._error_request(err)
        else:
            return self._after_request(*get_values_from_response(response))

    def post_replicas_failed(self, collectors=[], learners=[]):
        try:
            data = {
                "namespace": self.__namespace,
                "coordinator": self.__my_name,
                "collectors": collectors,
                "learners": learners,
            }
            response = self.__http_engine.request('POST', self.__prefix_with_api_version('/replicas/failed'), data=data)
        except RequestException as err:
            return self._error_request(err)
        else:
            return self._after_request(*get_values_from_response(response))

    def delete_replicas(self, n_collectors=0, n_learners=0):
        try:
            data = {
                "namespace": self.__namespace,
                "coordinator": self.__my_name,
                "collectors": {
                    "replicas": n_collectors,
                },
                "learners": {
                    "replicas": n_learners,
                }
            }
            response = self.__http_engine.request('DELETE', self.__prefix_with_api_version('/replicas'), data=data)
        except RequestException as err:
            return self._error_request(err)
        else:
            return self._after_request(*get_values_from_response(response))

    def _after_request(
            self, status_code: int, success: bool, code: int, message: Optional[str], data: Optional[Mapping[str, Any]]
    ) -> Any:
        return success, code, message, data

    def _error_request(self, error: RequestException) -> Any:
        # raise error
        raise RequestException
