from typing import Callable, Mapping, Any, Optional

from requests import RequestException

_BEFORE_HOOK_TYPE = Callable[..., Mapping[str, Any]]
_AFTER_HOOK_TYPE = Callable[[int, bool, int, Optional[str], Optional[Mapping[str, Any]]], Any]
_ERROR_HOOK_TYPE = Callable[[RequestException], Any]
