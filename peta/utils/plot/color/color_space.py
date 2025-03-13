from typing import Tuple


def hex_to_rgb(hex: str) -> Tuple[int, int, int]:
    """
    Converts a hexadecimal color code to a tuple of
    integers corresponding to its RGB components.

    Examples:

        >>> hex_to_rgb('FFA501') # (255, 165, 1)

    Args:
        hex (str): '#000000' or '000000'

    Returns:
        Tuple[int,int,int]: color in RGB color space.
    """
    if hex[0] == "#":
        hex = hex[1:]
    assert len(hex) == 6

    return tuple(int(hex[i : i + 2], 16) for i in (0, 2, 4))
