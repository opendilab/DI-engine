import importlib
import logging
from typing import List

global ceph_flag, linklink_flag
ceph_flag, linklink_flag = True, True


def try_import_ceph():
    """
    Try import ceph module, if failed, return None

    Returns:
        module: imported module, or None when ceph not found
    """
    global ceph_flag
    try:
        import ceph
    except ModuleNotFoundError as e:
        if ceph_flag:
            logging.warning(
                "You have not installed ceph package! If you are not run locally and testing, "
                "ask coworker for help."
            )
        ceph = None
        ceph_flag = False
    return ceph


def try_import_link():
    global linklink_flag
    """
    Try import linklink module, if failed, import nervex.tests.fake_linklink instead

    Returns:
        module: imported module (may be fake_linklink)
    """
    try:
        import linklink as link
    except ModuleNotFoundError as e:
        if linklink_flag:
            logging.warning(
                "You have not installed linklink package! If you are not run locally and testing, "
                "ask coworker for help. We will run a fake linklink."
                "Refer to nervex.tests.fake_linklink.py for details."
            )
        from nervex.tests.fake_linklink import link
        linklink_flag = False
    return link


def import_module(modules: List[str]) -> None:
    """
    Import several module as a list
    Args:
        modules (list): List of module names
    """
    for name in modules:
        importlib.import_module(name)
