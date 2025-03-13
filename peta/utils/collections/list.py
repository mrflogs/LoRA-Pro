from collections import defaultdict
from typing import Any, Callable, Iterable


def sorted_list(iter: Iterable) -> list:
    """
    convert iterable object `iter` to sorted list
    """
    ans = list(iter)
    ans.sort()
    return ans


def list_ignore(iter: Iterable, ignore=None) -> list:
    """
    返回一个新列表，丢弃ignore指定的元素
    Args:
        iter (Iterable): _description_
        ignore (_type_, optional): _description_. Defaults to None.
    Returns:
        list
    """
    if ignore is None:
        return [item for item in iter if item is not None]
    else:
        return [item for item in iter if item in ignore]


def group_by(lst: Iterable, fn: Callable[[Any], Any]):
    R"""
    Groups the elements of a list based on the given function.

    Examples:

        >>> from math import floor

        >>> group_by([6.1, 4.2, 6.3], floor)  # {4: [4.2], 6: [6.1, 6.3]}
        >>> group_by(["one", "two", "three"], len)  # {3: ['one', 'two'], 5: ['three']}

    Returns:
        defaultdict
    """
    d = defaultdict(list)
    for el in lst:
        d[fn(el)].append(el)
    return dict(d)
