from copy import deepcopy
from random import randint
from typing import List


def shuffle(lst: List):
    R"""
    Randomizes the order of the values of an list, returning a new list.

    Examples:

        >>> foo = [1, 2, 3]
        >>> shuffle(foo) # [2, 3, 1], foo = [1, 2, 3]
    """
    temp_lst = deepcopy(lst)
    m = len(temp_lst)
    while m:
        m -= 1
        i = randint(0, m)
        temp_lst[m], temp_lst[i] = temp_lst[i], temp_lst[m]
    return temp_lst


def shuffle_inplace(lst: List):
    m = len(lst)
    while m:
        m -= 1
        i = randint(0, m)
        lst[m], lst[i] = lst[i], lst[m]
    return lst
