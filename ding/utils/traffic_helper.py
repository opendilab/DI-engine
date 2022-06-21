from collections import deque
from typing import TYPE_CHECKING, Union
import time
import os
import json
from ditk import logging
from random import random
import pandas as pd
import ding.utils

if TYPE_CHECKING:
    from ding.framework import Parallel


class Traffic:

    def __init__(self) -> None:
        self._file = None
        self._router = None
        self._data = None

    def set_config(
        self,
        is_writer: bool = False,
        file_path: str = None,
        maxlen: int = 10000,
        router: "Parallel" = None
    ) -> "Traffic":
        """
        Overview:
            Enable and change the configuration of a Traffic instance.
        Arguments:
            - is_writer (:obj:`bool`): To enable permisson for file writing with data in memory.
            - file_path (:obj:`str`): File path that offline data are to be saved at. \
                In local mode, it must be set sucessfully for at least once. \
                In remote mode, no need to set it for worker, but is necessary for master. \
            - maxlen (:obj:`int`): Max data size in memory.
            - router (:obj:`Parallel`): To enable remote mode.
        """

        if is_writer:
            assert maxlen > 0

        if self._file:
            logging.warn("Configuration failure: file handle existed.")
        elif file_path:
            if not is_writer:
                logging.warn("The traffic is not set as writer.")
            else:
                try:
                    if not os.path.exists(os.path.dirname(file_path)):
                        os.makedirs(os.path.dirname(file_path))
                    self._file = open(file_path, "w+")
                except IOError as error:
                    self._file = None
                    logging.error("Invalid file path.")
                    raise

        if self._router and self._router.is_active:
            logging.warn("Configuration failure: parallel router existed.")
        elif router and router.is_active:
            self._router = router
            if self._file:
                router.on("_Traffic_", self._on_msg)

        if self._data is not None:
            logging.warn("Configuration failure: data existed.")
        elif is_writer:
            self._data = deque(maxlen=maxlen)
            self._max_records = 0
            self._cache = {}

        return self

    def record(self, info: dict = None, **kwargs) -> None:
        """
        Overview:
            Record information message of a dictionary form, and may save it in local file, or emit it to master, \
                or save it in memory for online analysis. \
                Only basic string or numeric data type is allowed to be saved.
        Arguments:
            - info (:obj:`dict`): information message of dictionary form.
            - kwargs (:obj:`any`): any information needed to be recorded.
        """
        dict_to_record = None
        if info:
            if kwargs:
                dict_to_record = {**info, **kwargs}
            else:
                dict_to_record = {**info}
        else:
            if kwargs:
                dict_to_record = {**kwargs}
            else:
                dict_to_record = {}

        if "__time" not in dict_to_record:
            dict_to_record.update({"__time": time.time()})

        illegel_keys = []
        for key in dict_to_record:
            if type(dict_to_record[key]) not in [str, int, float, complex]:
                illegel_keys.append(key)
        for key in illegel_keys:
            dict_to_record.pop(key, None)

        msg = json.dumps(dict_to_record)
        if self._file:
            self._file.write(msg + "\n")
        elif self._router and self._router.is_active:
            self._router.emit("_Traffic_", dict_to_record)

        if self._data is not None:
            self._data.append(dict_to_record)
            self._max_records += 1

    @property
    def df(self) -> pd.DataFrame:
        """
        Overview:
            Obtain current data in memory.
        """
        if self._data is not None:
            df = self._cache.get(self._max_records)
            if df is not None:
                return df
            df = pd.DataFrame(self._data)
            self._cache = {self._max_records: df}
            return df
        else:
            return None

    def close(self) -> None:
        """
        Overview:
            Safely close the module.
        """
        if self._router:
            self._router.off("_Traffic_")
            self._router = None
        if self._file:
            self._file.close()
            self._file = None
        if self._data is not None:
            self._cache.clear()
            self._data.clear()
            self._data = None
            self._max_records = 0

    def _on_msg(self, info: object, *args, **kwargs) -> None:
        """
        Overview:
            Listen for RPC from non-master instance.
            *** Private method ***
        """
        self.record(info)

    def __del__(self) -> None:
        self.close()


traffic = Traffic()
