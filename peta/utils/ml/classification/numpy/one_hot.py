import numpy as np


def to_categorical(Y: np.ndarray, num_classes: int, dtype=np.int64) -> np.ndarray:
    """
    to one-hot encoding

    Args:
        Y (np.ndarray): shape of (n_samples,)
        num_classes (int): _description_
        dtype: type of elements in return array

    Returns:
        np.ndarray: array of shape (nsamples, num_classes)
    """
    assert Y.ndim == 1, "Y must be 1-dimensional"

    n = Y.shape[0]
    Y_one_hot = np.zeros((n, num_classes), dtype=dtype)
    for i in range(n):
        Y_one_hot[i, Y[i]] = 1
    return Y_one_hot


def from_categorical(Y_one_hot: np.ndarray) -> np.ndarray:
    """
    from one-hot encoding

    Args:
        Y_one_hot (np.ndarray): shape of (nsamples, num_classes)

    Returns:
        np.ndarray: array of shape (nsamples,)
    """
    assert Y_one_hot.ndim == 2, "Y_one_hot must be 2-dimensional"

    n = Y_one_hot.shape[0]
    Y = np.zeros(n, dtype=Y_one_hot.dtype)
    for i in range(n):
        Y[i] = np.argmax(Y_one_hot[i])
    return Y
