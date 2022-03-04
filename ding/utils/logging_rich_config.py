import platform
import os
import sys
from typing import Optional
import logging
from rich.logging import RichHandler
from rich import console


def enable_rich_handler(level: int = logging.INFO) -> None:
    """
    Overview:
        Enable rich handler decoration to logger. Default logging.StreamHandler will be replaced. \
            Default terminal size is automatic dectected for logging message. \
            If no terminal detected, a default value is set for Rich handeler.
    Arguments:
        - level (:obj:`int`): Logger Level for Rich handeler, default set to ``logging.INFO``.
    """

    width: Optional[int] = None

    if platform.system() == "Windows":  # pragma: no cover
        try:
            width, _ = os.get_terminal_size()
        except OSError:  # Probably not a terminal
            pass
    else:
        try:
            #Try to get terminal size from the standard input file descriptor.
            width, _ = os.get_terminal_size(sys.__stdin__.fileno())
        except (AttributeError, ValueError, OSError):
            # AttributeError for access non-exist attribution.
            # ValueError for illegal size data format, such as expecting 2 varianbles but got 0.
            # OSError for inappropriate ioctl for the device, such as running in kubenetes.
            try:
                #Try to get terminal size from the standard output file descriptor.
                width, _ = os.get_terminal_size(sys.__stdout__.fileno())
            except (AttributeError, ValueError, OSError):
                pass

    # get_terminal_size can report 0, 0 if run from pseudo-terminal
    width = width or 285

    root = logging.getLogger()
    other_handlers = []

    if root.handlers:
        for handler in root.handlers[:]:
            root.removeHandler(handler)
            if type(handler) is logging.StreamHandler or type(handler) is RichHandler:
                handler.close()
            else:
                other_handlers.append(handler)

    other_handlers.append(RichHandler(console=console.Console(width=width)))

    logging.basicConfig(level=level, format="%(message)s", datefmt="[%Y-%m-%d %H:%M:%S]", handlers=other_handlers)


def disable_rich_handler(level: int = logging.INFO) -> None:
    """
    Overview:
        Disable rich handler decoration to logger. RichHandler will be replaced by logging.StreamHandler.
    Arguments:
        - level (:obj:`int`): Logger Level for Rich handeler, default set to ``logging.INFO``.
    """

    root = logging.getLogger()
    other_handlers = []

    if root.handlers:
        for handler in root.handlers[:]:
            root.removeHandler(handler)
            if type(handler) is logging.StreamHandler or type(handler) is RichHandler:
                handler.close()
            else:
                other_handlers.append(handler)

    other_handlers.append(logging.StreamHandler())

    logging.basicConfig(level=level, format="%(message)s", datefmt="[%Y-%m-%d %H:%M:%S]", handlers=other_handlers)
