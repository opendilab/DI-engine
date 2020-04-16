import numpy as np


def pfsp(win_rates, weighting):
    """
    Overview: prioritized fictitious self-play algorithm
    Arguments:
        - win_rates (:obj:`np.array`): a numpy array of win rates for(), shape(N)
        - weighting (:obj:`str`): pfsp weighting function type, refer to the below weighting_func
    Returns:
        - probs (:obj:`np.array`): a numpy array of the corresponding probability of each element is selected, shape(N)
    """
    weighting_func = {
        'squared': lambda x: (1 - x)**2,
        'variance': lambda x: x * (1 - x),
    }
    if weighting in weighting_func.keys():
        fn = weighting_func[weighting]
    else:
        return KeyError("invalid weighting arg: {} in pfsp".format(weighting))

    assert isinstance(win_rates, np.ndarray)
    # all zero win rates case
    if win_rates.sum() < 1e-8:
        return np.ones_like(win_rates) / len(win_rates)
    fn_win_rates = fn(win_rates)
    probs = fn_win_rates / fn_win_rates.sum()
    return probs
