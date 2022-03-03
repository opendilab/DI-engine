import platform
import os
import sys
from typing import Optional
import logging
from rich.logging import RichHandler
from rich import console


def enable_rich_handler(
        level: int = logging.INFO, terminal_width: Optional[int] = None, terminal_height: Optional[int] = None
) -> None:
    r'''
    Overview:
        Enable rich handler decoration to logger. Default logging.StreamHandler will be replaced.
    Arguments:
        - level (:obj:`int`): Logger Level for Rich handeler, default set to ``logging.INFO``.
        - terminal_width (:obj:`int`): The designed terminal width for logging message.
        - terminal_height (:obj:`int`): The designed terminal height for logging message,  
         default set to ``None`` for an automatic dectection is activated.
         If no terminal detected, a default value is set for Rich handeler.
    '''

    width: Optional[int] = None
    height: Optional[int] = None

    if platform.system() == "Windows":  # pragma: no cover
        try:
            width, height = os.get_terminal_size()
        except OSError:  # Probably not a terminal
            pass
    else:
        try:
            width, height = os.get_terminal_size(sys.__stdin__.fileno())
        except (AttributeError, ValueError, OSError):
            try:
                width, height = os.get_terminal_size(sys.__stdout__.fileno())
            except (AttributeError, ValueError, OSError):
                pass

    # get_terminal_size can report 0, 0 if run from pseudo-terminal
    width = terminal_width or width or 285
    height = terminal_height or height or 25

    root = logging.getLogger()
    other_handlers = []

    if root.handlers:
        for handler in root.handlers:
            root.removeHandler(handler)
            if isinstance(handler,
                          logging.StreamHandler) and not isinstance(handler, logging.FileHandler) or isinstance(
                              handler, RichHandler):
                handler.close()
            else:
                other_handlers.append(handler)

    other_handlers.append(RichHandler(console=console.Console(width=width)))

    logging.basicConfig(level=level, format="%(message)s", datefmt="[%Y-%m-%d %H:%M:%S]", handlers=other_handlers)


def disable_rich_handler(level: int = logging.INFO) -> None:
    r'''
    Overview:
        Disable rich handler decoration to logger. RichHandler will be replaced by logging.StreamHandler.
    Arguments:
        - level (:obj:`int`): Logger Level for Rich handeler, default set to ``logging.INFO``.
    '''
    root = logging.getLogger()
    other_handlers = []

    if root.handlers:
        for handler in root.handlers:
            root.removeHandler(handler)
            if isinstance(handler,
                          logging.StreamHandler) and not isinstance(handler, logging.FileHandler) or isinstance(
                              handler, RichHandler):
                handler.close()
            else:
                other_handlers.append(handler)

    other_handlers.append(logging.StreamHandler())

    logging.basicConfig(level=level, format="%(message)s", datefmt="[%Y-%m-%d %H:%M:%S]", handlers=other_handlers)
