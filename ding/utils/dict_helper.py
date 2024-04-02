from easydict import EasyDict


def convert_easy_dict_to_dict(easy_dict: EasyDict) -> dict:
    """
    Overview:
        Convert an EasyDict object to a dict object recursively.
    Arguments:
        - easy_dict (:obj:`EasyDict`): The EasyDict object to be converted.
    Returns:
        - dict: The converted dict object.
    """
    return {k: convert_easy_dict_to_dict(v) if isinstance(v, EasyDict) else v for k, v in easy_dict.items()}
