# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from typing import Tuple

import torch
from deepspeed.accelerator import get_accelerator

TOLERANCES = None


def get_tolerances():
    global TOLERANCES
    if TOLERANCES is None:
        TOLERANCES = {torch.float32: (5e-4, 5e-5), torch.float16: (3e-2, 2e-3)}
        if get_accelerator().is_bf16_supported():
            # Note: BF16 tolerance is higher than FP16 because of the lower precision (7 (+1) bits vs
            # 10 (+1) bits)
            TOLERANCES[torch.bfloat16] = (4.8e-1, 3.2e-2)
    return TOLERANCES


DTYPES = None


def get_dtypes(include_float=True):
    global DTYPES
    if DTYPES is None:
        DTYPES = [torch.float16, torch.float32] if include_float else [torch.float16]
        try:
            if get_accelerator().is_bf16_supported():
                DTYPES.append(torch.bfloat16)
        except (AssertionError, AttributeError):
            pass
    return DTYPES


def allclose(x, y, tolerances: Tuple[int, int] = None):
    assert x.dtype == y.dtype
    if tolerances is None:
        rtol, atol = get_tolerances()[x.dtype]
    else:
        rtol, atol = tolerances
    return torch.allclose(x, y, rtol=rtol, atol=atol)
