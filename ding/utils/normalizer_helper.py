import numpy as np


class DatasetNormalizer:
    """
    Overview:
        The `DatasetNormalizer` class provides functionality to normalize and unnormalize data in a dataset.
        It takes a dataset as input and applies a normalizer function to each key in the dataset.

    Interfaces:
        ``__init__``, ``__repr__``, ``normalize``, ``unnormalize``.
    """

    def __init__(self, dataset: np.ndarray, normalizer: str, path_lengths: list = None):
        """
        Overview:
            Initialize the NormalizerHelper object.

        Arguments:
            - dataset (:obj:`np.ndarray`): The dataset to be normalized.
            - normalizer (:obj:`str`): The type of normalizer to be used. Can be a string representing the name of \
                the normalizer class.
            - path_lengths (:obj:`list`): The length of the paths in the dataset. Defaults to None.
        """
        dataset = flatten(dataset, path_lengths)

        self.observation_dim = dataset['observations'].shape[1]
        self.action_dim = dataset['actions'].shape[1]

        if isinstance(normalizer, str):
            normalizer = eval(normalizer)

        self.normalizers = {}
        for key, val in dataset.items():
            try:
                self.normalizers[key] = normalizer(val)
            except:
                print(f'[ utils/normalization ] Skipping {key} | {normalizer}')
            # key: normalizer(val)
            # for key, val in dataset.items()

    def __repr__(self) -> str:
        """
        Overview:
            Returns a string representation of the NormalizerHelper object. \
            The string representation includes the key-value pairs of the normalizers \
            stored in the NormalizerHelper object.
        Returns:
            - ret (:obj:`str`):A string representation of the NormalizerHelper object.
        """
        string = ''
        for key, normalizer in self.normalizers.items():
            string += f'{key}: {normalizer}]\n'
        return string

    def normalize(self, x: np.ndarray, key: str) -> np.ndarray:
        """
        Overview:
            Normalize the input data using the specified key.

        Arguments:
            - x (:obj:`np.ndarray`): The input data to be normalized.
            - key (:obj`str`): The key to identify the normalizer.

        Returns:
            - ret (:obj:`np.ndarray`): The normalized value of the input data.
        """
        return self.normalizers[key].normalize(x)

    def unnormalize(self, x: np.ndarray, key: str) -> np.ndarray:
        """
        Overview:
            Unnormalizes the given value `x` using the specified `key`.

        Arguments:
            - x (:obj:`np.ndarray`): The value to be unnormalized.
            - key (:obj`str`): The key to identify the normalizer.

        Returns:
            - ret (:obj:`np.ndarray`): The unnormalized value.
        """
        return self.normalizers[key].unnormalize(x)


def flatten(dataset: dict, path_lengths: list) -> dict:
    """
    Overview:
        Flattens dataset of { key: [ n_episodes x max_path_length x dim ] } \
        to { key : [ (n_episodes * sum(path_lengths)) x dim ] }

    Arguments:
        - dataset (:obj:`dict`): The dataset to be flattened.
        - path_lengths (:obj:`list`): A list of path lengths for each episode.

    Returns:
        - flattened (:obj:`dict`): The flattened dataset.
    """
    flattened = {}
    for key, xs in dataset.items():
        assert len(xs) == len(path_lengths)
        if key == 'path_lengths':
            continue
        flattened[key] = np.concatenate([x[:length] for x, length in zip(xs, path_lengths)], axis=0)
    return flattened


class Normalizer:
    """
    Overview:
        Parent class, subclass by defining the `normalize` and `unnormalize` methods

    Interfaces:
        ``__init__``, ``__repr__``, ``normalize``, ``unnormalize``.
    """

    def __init__(self, X):
        """
        Overview:
            Initialize the Normalizer object.
        Arguments:
            - X (:obj:`np.ndarray`): The data to be normalized.
        """

        self.X = X.astype(np.float32)
        self.mins = X.min(axis=0)
        self.maxs = X.max(axis=0)

    def __repr__(self) -> str:
        """
        Overview:
            Returns a string representation of the Normalizer object.
        Returns:
            - ret (:obj:`str`): A string representation of the Normalizer object.
        """

        return (
            f"""[ Normalizer ] dim: {self.mins.size}\n    -: """
            f"""{np.round(self.mins, 2)}\n    +: {np.round(self.maxs, 2)}\n"""
        )

    def normalize(self, *args, **kwargs):
        """
        Overview:
            Normalize the input data.
        Arguments:
            - args (:obj:`list`): The arguments passed to the ``normalize`` function.
            - kwargs (:obj:`dict`): The keyword arguments passed to the ``normalize`` function.
        """

        raise NotImplementedError()

    def unnormalize(self, *args, **kwargs):
        """
        Overview:
            Unnormalize the input data.
        Arguments:
            - args (:obj:`list`): The arguments passed to the ``unnormalize`` function.
            - kwargs (:obj:`dict`): The keyword arguments passed to the ``unnormalize`` function.
        """

        raise NotImplementedError()


class GaussianNormalizer(Normalizer):
    """
    Overview:
        A class that normalizes data to zero mean and unit variance.

    Interfaces:
        ``__init__``, ``__repr__``, ``normalize``, ``unnormalize``.
    """

    def __init__(self, *args, **kwargs):
        """
        Overview:
            Initialize the GaussianNormalizer object.
        Arguments:
            - args (:obj:`list`): The arguments passed to the ``__init__`` function of the parent class, \
                i.e., the Normalizer class.
            - kwargs (:obj:`dict`): The keyword arguments passed to the ``__init__`` function of the parent class, \
                i.e., the Normalizer class.
        """

        super().__init__(*args, **kwargs)
        self.means = self.X.mean(axis=0)
        self.stds = self.X.std(axis=0)
        self.z = 1

    def __repr__(self) -> str:
        """
        Overview:
            Returns a string representation of the GaussianNormalizer object.
        Returns:
            - ret (:obj:`str`): A string representation of the GaussianNormalizer object.
        """

        return (
            f"""[ Normalizer ] dim: {self.mins.size}\n    """
            f"""means: {np.round(self.means, 2)}\n    """
            f"""stds: {np.round(self.z * self.stds, 2)}\n"""
        )

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """
        Overview:
            Normalize the input data.

        Arguments:
            - x (:obj:`np.ndarray`): The input data to be normalized.

        Returns:
            - ret (:obj:`np.ndarray`): The normalized data.
        """
        return (x - self.means) / self.stds

    def unnormalize(self, x: np.ndarray) -> np.ndarray:
        """
        Overview:
            Unnormalize the input data.

        Arguments:
            - x (:obj:`np.ndarray`): The input data to be unnormalized.

        Returns:
            - ret (:obj:`np.ndarray`): The unnormalized data.
        """
        return x * self.stds + self.means


class CDFNormalizer(Normalizer):
    """
    Overview:
        A class that makes training data uniform (over each dimension) by transforming it with marginal CDFs.

    Interfaces:
        ``__init__``, ``__repr__``, ``normalize``, ``unnormalize``.
    """

    def __init__(self, X):
        """
        Overview:
            Initialize the CDFNormalizer object.
        Arguments:
            - X (:obj:`np.ndarray`): The data to be normalized.
        """

        super().__init__(atleast_2d(X))
        self.dim = self.X.shape[1]
        self.cdfs = [CDFNormalizer1d(self.X[:, i]) for i in range(self.dim)]

    def __repr__(self) -> str:
        """
        Overview:
            Returns a string representation of the CDFNormalizer object.
        Returns:
            - ret (:obj:`str`): A string representation of the CDFNormalizer object.
        """

        return f'[ CDFNormalizer ] dim: {self.mins.size}\n' + '    |    '.join(
            f'{i:3d}: {cdf}' for i, cdf in enumerate(self.cdfs)
        )

    def wrap(self, fn_name: str, x: np.ndarray) -> np.ndarray:
        """
        Overview:
            Wraps the given function name and applies it to the input data.

        Arguments:
            - fn_name (:obj:`str`): The name of the function to be applied.
            - x (:obj:`np.ndarray`): The input data.

        Returns:
            - ret: The output of the function applied to the input data.
        """
        shape = x.shape
        # reshape to 2d
        x = x.reshape(-1, self.dim)
        out = np.zeros_like(x)
        for i, cdf in enumerate(self.cdfs):
            fn = getattr(cdf, fn_name)
            out[:, i] = fn(x[:, i])
        return out.reshape(shape)

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """
        Overview:
            Normalizes the input data.

        Arguments:
            - x (:obj:`np.ndarray`): The input data.

        Returns:
            - ret (:obj:`np.ndarray`): The normalized data.
        """
        return self.wrap('normalize', x)

    def unnormalize(self, x: np.ndarray) -> np.ndarray:
        """
        Overview:
            Unnormalizes the input data.

        Arguments:
            - x (:obj:`np.ndarray`): The input data.

        Returns:
            - ret (:obj:`np.ndarray`):: The unnormalized data.
        """
        return self.wrap('unnormalize', x)


class CDFNormalizer1d:
    """
    Overview:
        CDF normalizer for a single dimension. This class provides methods to normalize and unnormalize data \
        using the Cumulative Distribution Function (CDF) approach.
    Interfaces:
        ``__init__``, ``__repr__``, ``normalize``, ``unnormalize``.
    """

    def __init__(self, X: np.ndarray):
        """
        Overview:
            Initialize the CDFNormalizer1d object.
        Arguments:
            - X (:obj:`np.ndarray`): The data to be normalized.
        """

        import scipy.interpolate as interpolate
        assert X.ndim == 1
        self.X = X.astype(np.float32)
        if self.X.max() == self.X.min():
            self.constant = True
        else:
            self.constant = False
            quantiles, cumprob = empirical_cdf(self.X)
            self.fn = interpolate.interp1d(quantiles, cumprob)
            self.inv = interpolate.interp1d(cumprob, quantiles)

            self.xmin, self.xmax = quantiles.min(), quantiles.max()
            self.ymin, self.ymax = cumprob.min(), cumprob.max()

    def __repr__(self) -> str:
        """
        Overview:
            Returns a string representation of the CDFNormalizer1d object.
        """

        return (f'[{np.round(self.xmin, 2):.4f}, {np.round(self.xmax, 2):.4f}')

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """
        Overview:
            Normalize the input data.

        Arguments:
            - x (:obj:`np.ndarray`): The data to be normalized.

        Returns:
            - ret (:obj:`np.ndarray`): The normalized data.
        """
        if self.constant:
            return x

        x = np.clip(x, self.xmin, self.xmax)
        # [ 0, 1 ]
        y = self.fn(x)
        # [ -1, 1 ]
        y = 2 * y - 1
        return y

    def unnormalize(self, x: np.ndarray, eps: float = 1e-4) -> np.ndarray:
        """
        Overview:
            Unnormalize the input data.

        Arguments:
            - x (:obj:`np.ndarray`): The data to be unnormalized.
            - eps (:obj:`float`): A small value used for numerical stability. Defaults to 1e-4.

        Returns:
            - ret (:obj:`np.ndarray`): The unnormalized data.
        """
        # [ -1, 1 ] --> [ 0, 1 ]
        if self.constant:
            return x

        x = (x + 1) / 2.

        if (x < self.ymin - eps).any() or (x > self.ymax + eps).any():
            print(
                f"""[ dataset/normalization ] Warning: out of range in unnormalize: """
                f"""[{x.min()}, {x.max()}] | """
                f"""x : [{self.xmin}, {self.xmax}] | """
                f"""y: [{self.ymin}, {self.ymax}]"""
            )

        x = np.clip(x, self.ymin, self.ymax)

        y = self.inv(x)
        return y


def empirical_cdf(sample: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Overview:
        Compute the empirical cumulative distribution function (CDF) of a given sample.

    Arguments:
        - sample (:obj:`np.ndarray`): The input sample for which to compute the empirical CDF.

    Returns:
        - quantiles (:obj:`np.ndarray`): The unique values in the sample.
        - cumprob (:obj:`np.ndarray`): The cumulative probabilities corresponding to the quantiles.

    References:
        - Stack Overflow: https://stackoverflow.com/a/33346366
    """

    # find the unique values and their corresponding counts
    quantiles, counts = np.unique(sample, return_counts=True)

    # take the cumulative sum of the counts and divide by the sample size to
    # get the cumulative probabilities between 0 and 1
    cumprob = np.cumsum(counts).astype(np.double) / sample.size

    return quantiles, cumprob


def atleast_2d(x: np.ndarray) -> np.ndarray:
    """
    Overview:
        Ensure that the input array has at least two dimensions.

    Arguments:
        - x (:obj:`np.ndarray`): The input array.

    Returns:
        - ret (:obj:`np.ndarray`): The input array with at least two dimensions.
    """
    if x.ndim < 2:
        x = x[:, None]
    return x


class LimitsNormalizer(Normalizer):
    """
    Overview:
        A class that normalizes and unnormalizes values within specified limits. \
        This class maps values within the range [xmin, xmax] to the range [-1, 1].

    Interfaces:
        ``__init__``, ``__repr__``, ``normalize``, ``unnormalize``.
    """

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """
        Overview:
            Normalizes the input values.

        Argments:
            - x (:obj:`np.ndarray`): The input values to be normalized.

        Returns:
            - ret (:obj:`np.ndarray`): The normalized values.

        """
        # [ 0, 1 ]
        x = (x - self.mins) / (self.maxs - self.mins)
        # [ -1, 1 ]
        x = 2 * x - 1
        return x

    def unnormalize(self, x: np.ndarray, eps: float = 1e-4) -> np.ndarray:
        """
        Overview:
            Unnormalizes the input values.

        Arguments:
            - x (:obj:`np.ndarray`): The input values to be unnormalized.
            - eps (:obj:`float`): A small value used for clipping. Defaults to 1e-4.

        Returns:
            - ret (:obj:`np.ndarray`): The unnormalized values.

        """
        if x.max() > 1 + eps or x.min() < -1 - eps:
            # print(f'[ datasets/mujoco ] Warning: sample out of range | ({x.min():.4f}, {x.max():.4f})')
            x = np.clip(x, -1, 1)

        # [ -1, 1 ] --> [ 0, 1 ]
        x = (x + 1) / 2.

        return x * (self.maxs - self.mins) + self.mins
