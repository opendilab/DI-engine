import logging
from rich.logging import RichHandler

logging.basicConfig(level=logging.NOTSET, format="%(message)s", datefmt="[%X]", handlers=[RichHandler()], force=True)
