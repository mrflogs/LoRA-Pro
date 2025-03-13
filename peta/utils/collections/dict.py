from copy import deepcopy
from typing import Dict, Iterable, List, Tuple, Union


def dict_to_tuple(d: Dict):
    """
    Convert a dictionary to a tuple of key-value pairs.
    """
    return tuple(d.items())


def dict_get(d: dict, keys: Iterable[str], default=None):
    return [d.get(k, default) for k in keys]


def dict_map(f, d: dict, *, max_level: int = -1, skip_levels=0, inplace=False):
    """对d的每一个元素作用f，并返回一个新字典
    Args:
        f (_type_): function
        d (dict): 原字典
        max_level (int, optional): 作用的深度， -1 表示无限深. Defaults to -1.
        skip_levels (int, optional): 跳过的层数. Defaults to 0.
        inplace (bool, optional): Defaults to False.
    Returns:
        dict: 转换后的字典
    """
    if not isinstance(d, dict):
        raise TypeError("dict_map: d must be a dict")

    if inplace:
        ans = d
    else:
        ans = deepcopy(d)

    def dict_map_impl(from_dict, to_dict, level):
        if level == max_level:
            return
        for k in from_dict.keys():
            if isinstance(from_dict[k], dict):
                dict_map_impl(from_dict[k], to_dict[k], level + 1)
            else:
                if level < skip_levels:
                    continue
                else:
                    to_dict[k] = f(from_dict[k])

    dict_map_impl(d, ans, 0)
    return ans
