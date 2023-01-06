import numpy as np


def get_seq_reverse_complement(seq, num_reverse_compliment_dict):
    """[summary]
    https://stackoverflow.com/questions/55949809/efficiently-replace-elements-in-array-based-on-dictionary-numpy-python

    Returns:
        [type]: [description]
    """

    k = np.array(list(num_reverse_compliment_dict.keys()))
    v = np.array(list(num_reverse_compliment_dict.values()))

    reverse_seq = seq[::-1]
    mapping_ar = np.zeros(k.max() + 1, dtype=v.dtype)
    mapping_ar[k] = v
    return mapping_ar[reverse_seq]


def dummy_func(x):
    """
    Dummy function, returns its input. This seems useless, but it greatly
    simplifies the code of the dataloader, as it assumes that all inputs and
    outputs have a transform function, so those who do not need any transforms
    will pass through this.

    Args:
        x ([type]): [description]

    Returns:
        [type]: [description]
    """
    return x


def separate_elemets(y):
    """Splits an NxM array into a list of M Nx1 array. This is important for
    the model losses

    Args:
        y ([type]): [description]

    Returns:
        [type]: [description]
    """
    return [y[:, i] for i in range(y.shape[1])]
