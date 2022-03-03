import platform
import os
import sys
from typing import Optional
import logging
from rich.logging import RichHandler
from rich import console

WINDOWS = platform.system() == "Windows"

width: Optional[int] = None
height: Optional[int] = None

if WINDOWS:  # pragma: no cover
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
width = width or 285
height = height or 25

root = logging.getLogger()
if root.handlers:
    for handler in root.handlers:
        root.removeHandler(handler)

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%Y-%m-%d %H:%M:%S]",
    handlers=[RichHandler(console=console.Console(width=width))]
)
