import os
from pathlib import Path
from typing import List

import torch
from multipledispatch import dispatch


@dispatch(str)
def listdir_fullpath(dir: str) -> List[str]:
    """list directory `dir`, return fullpaths

    Args:
        dir (str): directory name

    Returns:
        List[str]: a list of fullpaths
    """
    assert os.path.isdir(dir), "Argument 'dir' must be a Directory"
    names = os.listdir(dir)
    return [os.path.join(dir, name) for name in names]


@dispatch(Path)
def listdir_fullpath(dir: Path) -> List[Path]:
    """list directory `dir`, return fullpaths

    Args:
        dir (Path): _description_

    Returns:
        List[Path]: _description_
    """
    names = dir.iterdir()
    return [dir / i for i in names]
