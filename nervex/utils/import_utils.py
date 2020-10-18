import importlib
import logging
from typing import List, NoReturn


def try_import_ceph():
    """
    Try import ceph module, if failed, return None

    Returns:
        module: imported module, or None when ceph not found
    """
    try:
        import ceph
    except ModuleNotFoundError as e:
        ceph = None
        logging.warning(
            "You have not installed ceph package! If you are not run locally and testing, "
            "ask coworker for help."
        )

    return ceph


def try_import_link():
    """
    Try import linklink module, if failed, import nervex.tests.fake_linklink instead

    Returns:
        module: imported module (may be fake_linklink)
    """
    try:
        import linklink as link
    except ModuleNotFoundError as e:
        from nervex.tests.fake_linklink import link
        logging.warning(
            "You have not installed linklink package! If you are not run locally and testing, "
            "ask coworker for help. We will run a fake linklink for you. "
            "Refer to nervex.tests.fake_linklink.py for details."
        )

    return link


def import_module(modules: List[str]) -> NoReturn:
    """
    Import several module as a list
    Args:
        modules (list): List of module names
    """
    for name in modules:
        importlib.import_module(name)
