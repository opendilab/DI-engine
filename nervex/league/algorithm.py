import numpy as np


def pfsp(win_rates: np.ndarray, weighting: str) -> np.ndarray:
    """
    Overview:
        Prioritized Fictitious Self-Play algorithm.
        Process win_rates with a weighting function to get priority, then calculate the selection probability of each.
    Arguments:
        - win_rates (:obj:`np.ndarray`): a numpy ndarray of win rates between one player and N opponents, shape(N)
        - weighting (:obj:`str`): pfsp weighting function type, refer to ``weighting_func`` below
    Returns:
        - probs (:obj:`np.ndarray`): a numpy ndarray of probability at which one element is selected, shape(N)
    """
    weighting_func = {
        'squared': lambda x: (1 - x) ** 2,
        'variance': lambda x: x * (1 - x),
    }
    if weighting in weighting_func.keys():
        fn = weighting_func[weighting]
    else:
        raise KeyError("invalid weighting arg: {} in pfsp".format(weighting))

    assert isinstance(win_rates, np.ndarray)
    assert win_rates.shape[0] >= 1, win_rates.shape
    # all zero win rates case, return uniform selection prob
    if win_rates.sum() < 1e-8:
        return np.full_like(win_rates, 1.0 / len(win_rates))
    fn_win_rates = fn(win_rates)
    probs = fn_win_rates / fn_win_rates.sum()
    return probs


def uniform(win_rates: np.ndarray) -> np.ndarray:
    """
    Overview:
        Uniform opponent selection algorithm. Select an opponent uniformly, regardless of historical win rates.
    Arguments:
        - win_rates (:obj:`np.ndarray`): a numpy ndarray of win rates between one player and N opponents, shape(N)
    Returns:
        - probs (:obj:`np.ndarray`): a numpy ndarray of uniform probability, shape(N)
    """
    return np.full_like(win_rates, 1.0 / len(win_rates))
