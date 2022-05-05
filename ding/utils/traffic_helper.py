from collections import deque
from typing import TYPE_CHECKING
import time
import os
import json
import logging
from random import random
import pandas as pd
import ding.utils

if TYPE_CHECKING:
    from ding.framework import Parallel


class Traffic:

    def __init__(self) -> None:
        pass

    def config(
            self,
            file_path: str = None,
            online: bool = False,
            maxlen: int = 10000,
            router: "Parallel" = None
    ) -> "Traffic":
        """
        Overview:
            Enable and change the configuration of a Traffic instance. 
        Arguments:
            - file_path (:obj:`str`): File path that offline data are to be saved at. \
                In local mode, it must be set sucessfully for at least once. \
                In remote mode, no need to set it for worker, but is necessary for master. \
            - online (:obj:`bool`): To enable online analysis and maintain data in memory.
            - maxlen (:obj:`int`): Max data size in memory.
            - router (:obj:`Parallel`): To enable remote mode.
        """

        if online and maxlen <= 0:
            logging.error("Illegal data size.")
            return self

        if not hasattr(self, "_file") or self._file is None:
            self._file = None
            self._has_file_writer = False
            self._file_path = None
            if file_path:
                try:
                    if not os.path.exists(os.path.dirname(file_path)):
                        os.makedirs(os.path.dirname(file_path))
                    self._file = open(file_path, "a+")
                    self._has_file_writer = True
                    self._file_path = file_path
                except IOError as error:
                    logging.error("Invalid file path.")
                    raise

        if not hasattr(self, "_router") or self._router is None:
            self._in_parallel = False
            self._router = None
            if router and router.is_active:
                self._in_parallel = True
                self._router = router
                if self._has_file_writer:
                    router.on("_Traffic_", self._on_msg)

        if not hasattr(self, "_has_data") or self._has_data is False:
            self._has_data = False
            if online:
                self._has_data = True
                self._data = deque(maxlen=maxlen)
                self._max_records = 0
                self._cache = {}

        return self

    def record(self, info: dict = None, **kwargs) -> None:
        """
        Overview:
            Record information message of a dictionary form, and may save it in local file, or emit it to master, \
                or save it in memory for online analysis.
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

        if not "__time" in dict_to_record:
            dict_to_record.update({"__time": time.time()})

        msg = json.dumps(dict_to_record)
        if self._has_file_writer and self._file:
            self._file.write(msg + "\n")

        if self._in_parallel and not self._has_file_writer:
            self._router.emit("_Traffic_", dict_to_record)

        if self._has_data:
            self._data.append(dict_to_record)
            self._max_records += 1

    @property
    def df(self) -> pd.DataFrame:
        """
        Overview:
            Obtain current data in memory.
        """
        if self._has_data:
            df = self._cache.get(self._max_records)
            if df is not None:
                return df
            df = pd.DataFrame(self._data)
            self._cache = {self._max_records: df}
            return df
        else:
            return None

    def close(self):
        """
        Overview:
            Safely close the module.
        """
        self._has_file_writer = False
        self._in_parallel = False
        if self._router:
            self._router.off("_Traffic_")
            self._router = None
        if self._file:
            self._file.close()
            self._file = None
        if self._has_data:
            self._cache.clear()
            self._data.clear()
            self._max_records = 0

    def _on_msg(self, info: object, *args, **kwargs):
        """
        Overview:
            Listen for RPC from non-master instance.
            *** Private method ***
        """
        self.record(info)

    def __del__(self):
        self.close()


traffic = Traffic()
