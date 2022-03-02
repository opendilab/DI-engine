import logging
from rich.logging import RichHandler

logging.basicConfig(
    level=logging.INFO, format="%(message)s", datefmt="[%Y-%m-%d %H:%M:%S]", handlers=[RichHandler()], force=True
)
