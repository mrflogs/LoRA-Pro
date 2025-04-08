# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import torch

import deepspeed
from deepspeed.accelerator import get_accelerator, is_current_accelerator_supported
from deepspeed.git_version_info import torch_info


def skip_on_arch(min_arch=7):
    if get_accelerator().device_name() == 'cuda':
        if torch.cuda.get_device_capability()[0] < min_arch:  #ignore-cuda
            pytest.skip(f"needs higher compute capability than {min_arch}")
    else:
        assert is_current_accelerator_supported()
        return


def skip_on_cuda(valid_cuda):
    split_version = lambda x: map(int, x.split('.')[:2])
    if get_accelerator().device_name() == 'cuda':
        CUDA_MAJOR, CUDA_MINOR = split_version(torch_info['cuda_version'])
        CUDA_VERSION = (CUDA_MAJOR * 10) + CUDA_MINOR
        if valid_cuda.count(CUDA_VERSION) == 0:
            pytest.skip(f"requires cuda versions {valid_cuda}")
    else:
        assert is_current_accelerator_supported()
        return


def bf16_required_version_check(accelerator_check=True):
    split_version = lambda x: map(int, x.split('.')[:2])
    TORCH_MAJOR, TORCH_MINOR = split_version(torch_info['version'])
    NCCL_MAJOR, NCCL_MINOR = split_version(torch_info['nccl_version'])
    CUDA_MAJOR, CUDA_MINOR = split_version(torch_info['cuda_version'])

    # Sometimes bf16 tests are runnable even if not natively supported by accelerator
    if accelerator_check:
        accelerator_pass = get_accelerator().is_bf16_supported()
    else:
        accelerator_pass = True

    torch_version_available = TORCH_MAJOR > 1 or (TORCH_MAJOR == 1 and TORCH_MINOR >= 10)
    cuda_version_available = CUDA_MAJOR >= 11
    nccl_version_available = NCCL_MAJOR > 2 or (NCCL_MAJOR == 2 and NCCL_MINOR >= 10)
    npu_available = get_accelerator().device_name() == 'npu'
    hpu_available = get_accelerator().device_name() == 'hpu'
    xpu_available = get_accelerator().device_name() == 'xpu'

    if torch_version_available and cuda_version_available and nccl_version_available and accelerator_pass:
        return True
    elif npu_available:
        return True
    elif hpu_available:
        return True
    elif xpu_available:
        return True
    else:
        return False


def required_amp_check():
    from importlib.util import find_spec
    if find_spec('apex') is None:
        return False
    else:
        return True


class no_child_process_in_deepspeed_io:

    def __enter__(self):
        # deepspeed_io defaults to creating a dataloader that uses a
        # multiprocessing pool. Our tests use pools and we cannot nest pools in
        # python. Therefore we're injecting this kwarg to ensure that no pools
        # are used in the dataloader.
        self.old_method = deepspeed.runtime.engine.DeepSpeedEngine.deepspeed_io

        def new_method(*args, **kwargs):
            kwargs["num_local_io_workers"] = 0
            return self.old_method(*args, **kwargs)

        deepspeed.runtime.engine.DeepSpeedEngine.deepspeed_io = new_method

    def __exit__(self, *_):
        deepspeed.runtime.engine.DeepSpeedEngine.deepspeed_io = self.old_method
