import numpy as np

def pad_left(n, times):
    return str(n).zfill(times)

def raise_if(condition, message, raise_type):
    if (condition):
        raise raise_type(message)

def remove(arr, indexes):
    assert isinstance(arr, list)
    
    if isinstance(indexes, int):
        arr.pop(indexes)
    
    if isinstance(indexes, list):
        for i in indexes:
            arr.pop(i)
    
def random_indexes(n, size, ignore=[]):
    """
    Returns a group of n indexes between 0 and size, avoiding ignore indexes.

    Params
    ------
    n number of indexes.
    size size of the vectors.
    ignore indexes to ignore.

    >>> random_indexes(1, 1)
    0
    >>> random_indexes(1, 2, [0])
    1
    >>> random_indexes(1, 2, [1])
    0
    >>> random_indexes(1, 3, [0, 1])
    2
    >>> random_indexes(1, 3, [0, 2])
    1
    >>> random_indexes(1, 3, [1, 2])
    0
    """
    indexes = [pos for pos in range(size) if pos not in ignore]
    
    if len(indexes) < n:
        print(ignore)
        print(indexes)
        print(size)
    
    assert len(indexes) >= n
    np.random.shuffle(indexes)

    if n == 1:
        return indexes[0]
    else:
        return indexes[:n]
    
def random_items(arr, n, size, ignore=[]):
    """
    Returns a group of n items between 0 and size, avoiding ignore indexes.

    Params
    ------
    n number of indexes.
    size size of the vectors.
    ignore indexes to ignore.

    >>> random_indexes(1, 1)
    0
    >>> random_indexes(1, 2, [0])
    1
    >>> random_indexes(1, 2, [1])
    0
    >>> random_indexes(1, 3, [0, 1])
    2
    >>> random_indexes(1, 3, [0, 2])
    1
    >>> random_indexes(1, 3, [1, 2])
    0
    """
    indexes = [pos for pos in range(size) if pos not in ignore]

    assert len(indexes) >= n
    np.random.shuffle(indexes)

    if n == 1:
        return arr[indexes[0]]
    else:
        return [arr[i] for i in indexes[:n]]