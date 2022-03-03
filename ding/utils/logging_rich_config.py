import platform
import os
import sys
from typing import Optional
import logging
from rich.logging import RichHandler
from rich import console


def enable_rich_handler(level: int = logging.INFO, terminal_width: Optional[int] = None) -> None:
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

    columns = os.environ.get("COLUMNS")
    if columns is not None and columns.isdigit():
        width = int(columns)
    lines = os.environ.get("LINES")
    if lines is not None and lines.isdigit():
        height = int(lines)

    # get_terminal_size can report 0, 0 if run from pseudo-terminal
    width = terminal_width or width or 285
    height = height or 25

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
