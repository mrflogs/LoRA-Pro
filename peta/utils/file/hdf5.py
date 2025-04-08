"""
This module contains utility functions for working with HDF5 files.

Functions:
- print_h5(group: h5py.Group, level=0, logger: Callable[[Any], Any] = print):
    Recursively prints the contents of an HDF5 group to the console.
"""
from typing import Any, Callable

import h5py


def print_h5(group: h5py.Group, level=0, logger: Callable[[Any], Any] = print):
    """print h5 group to consol.

    Args:
        group (h5py.Group): _description_
        level (int, optional): _description_. Defaults to 0.
        logger (Callable[[Any], Any]): print like functions
    """
    if not isinstance(group, h5py.Group):
        logger(level * "\t", group)
        return
    else:
        for key in group.keys():
            logger(level * "\t" + key + ":")
            subgroup = group[key]
            print_h5(subgroup, level + 1)
