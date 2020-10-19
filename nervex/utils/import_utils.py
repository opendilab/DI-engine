import importlib

global ceph_flag, linklink_flag
ceph_flag, linklink_flag = True, True


def try_import_ceph():
    global ceph_flag
    try:
        import ceph
    except ModuleNotFoundError as e:
        if ceph_flag:
            import logging
            logging.warning(
                "You have not installed ceph package! If you are not run locally and testing, "
                "ask coworker for help."
            )
        ceph = None
        ceph_flag = False
    return ceph


def try_import_link():
    global linklink_flag
    try:
        import linklink as link
    except ModuleNotFoundError as e:
        if linklink_flag:
            import logging
            logging.warning(
                "You have not installed linklink package! If you are not run locally and testing, "
                "ask coworker for help. We will run a fake linklink."
                "Refer to nervex.tests.fake_linklink.py for details."
            )
        from nervex.tests.fake_linklink import link
        linklink_flag = False
    return link


def import_module(names):
    for n in names:
        importlib.import_module(n)
