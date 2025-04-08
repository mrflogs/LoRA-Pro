"""
This module provides utility functions for working with JSON files.

Functions:
- print_json(j: dict, indent="  ", verbose: bool = False):
    Print an overview of a JSON file. The function takes a dictionary `j` as input, which represents the loaded JSON file.
    The `indent` parameter specifies the string used for indentation (default is two spaces).
    The `verbose` parameter controls whether to print the full content of lists of dictionaries or just their type and length.
"""
import json


def _is_list_of_dict(l) -> bool:
    if not isinstance(l, list):
        return False
    for i in l:
        if not isinstance(i, dict):
            return False
    return True


def _sprint_json_entry(obj):
    if isinstance(obj, str):
        return "string"
    elif isinstance(obj, float):
        return "float"
    elif isinstance(obj, int):
        return "int"
    elif isinstance(obj, list):
        if len(obj) > 0:
            return f"list[{_sprint_json_entry(obj[0])}]"
        else:
            return "list"
    else:
        return type(obj)


def print_json(j: dict, indent="  ", verbose: bool = False):
    R"""print an overview of json file

    Examples:
        >>> print_json(open('path_to_json', 'r'))

    Args:
        j (dict): loaded json file
        indent (str, optional): Defaults to '  '.
    """

    def _print_json(j: dict, level):
        def _sprint(s):
            return indent * level + s

        for k in j.keys():
            if isinstance(j[k], dict):
                print(_sprint(k) + ":")
                _print_json(j[k], level + 1)
            elif _is_list_of_dict(j[k]):
                if verbose:
                    print(_sprint(k) + ": [")
                    for i in range(len(j[k]) - 1):
                        _print_json(j[k][0], level + 2)
                        print(_sprint(f"{indent},"))
                    _print_json(j[k][-1], level + 2)
                    print(_sprint(f"{indent}]"))
                else:
                    print(_sprint(k) + ": [")
                    _print_json(j[k][0], level + 2)
                    print(_sprint(f"{indent}] ... {len(j[k])-1} more"))
            else:
                print(f"{_sprint(k)}: {_sprint_json_entry(j[k])}")

    _print_json(j, level=0)
