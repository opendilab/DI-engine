import logging
import pytest
import unittest
from rich.logging import RichHandler
from ding.utils import enable_rich_handler, disable_rich_handler


@pytest.mark.unittest
class TestRichLoggerConfiguration:

    def test_enable_rich_handler(self):

        enable_rich_handler()

        root = logging.getLogger()
        has_rich_handler = False
        has_multiple_rich_handler = False
        if root.handlers:
            for handler in root.handlers[:]:
                if type(handler) is logging.StreamHandler:
                    raise AssertionError("logging.StreamHandler should not exist.")

                if type(handler) is RichHandler:
                    if has_rich_handler:
                        has_multiple_rich_handler = True
                    has_rich_handler = True

        if has_multiple_rich_handler:
            raise AssertionError("Multiple rich handler is not allowed.")
        if not has_rich_handler:
            raise AssertionError("At least one rich handler should exist.")

    def test_disable_rich_handler(self):

        disable_rich_handler()

        root = logging.getLogger()
        has_stream_handler = False
        has_multiple_stream_handler = False
        if root.handlers:
            for handler in root.handlers[:]:
                if type(handler) is RichHandler:
                    raise AssertionError("RichHandler should not exist.")

                if type(handler) is logging.StreamHandler:
                    if has_stream_handler:
                        has_multiple_stream_handler = True
                    has_stream_handler = True

        if has_multiple_stream_handler:
            raise AssertionError("Multiple stream handler is not allowed.")
        if not has_stream_handler:
            raise AssertionError("At least one stream handler should exist.")
