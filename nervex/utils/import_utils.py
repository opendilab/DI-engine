def try_import_ceph():
    try:
        import ceph
    except ModuleNotFoundError as e:
        import logging
        logging.error(
            "You have not installed ceph package! If you are not run locally and testing, "
            "ask coworker for help."
        )
        ceph = None
    return ceph


def try_import_link():
    try:
        import linklink as link
    except ModuleNotFoundError as e:
        from nervex.tests.fake_linklink import link
        import logging

        logging.error(
            "You have not installed linklink package! If you are not run locally and testing, "
            "ask coworker for help. We will run a fake linklink for you. "
            "Refer to nervex.tests.fake_linklink.py for details."
        )
    return link
