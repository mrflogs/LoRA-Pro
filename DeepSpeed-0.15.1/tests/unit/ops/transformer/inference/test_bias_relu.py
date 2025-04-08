# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import torch
import deepspeed
from deepspeed.accelerator import get_accelerator
from deepspeed.ops.op_builder import InferenceBuilder
from .inference_test_utils import allclose, get_dtypes

if not deepspeed.ops.__compatible_ops__[InferenceBuilder.NAME]:
    pytest.skip("Inference ops are not available on this system", allow_module_level=True)

inference_module = None
torch_minor_version = None


def run_bias_relu_reference(activations, bias):
    # Expected behavior is that of casting to float32 internally
    return torch.nn.functional.relu(activations.to(torch.float32) + bias.to(torch.float32)).to(activations.dtype)


def run_bias_relu_ds(activations, bias):
    global inference_module
    if inference_module is None:
        inference_module = InferenceBuilder().load()
    if activations.dtype == torch.float16:
        return inference_module.bias_relu_fp16(activations, bias)
    elif activations.dtype == torch.bfloat16:
        return inference_module.bias_relu_bf16(activations, bias)
    else:
        return inference_module.bias_relu_fp32(activations, bias)


@pytest.mark.inference_ops
@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("sequence", [1, 128, 255])
@pytest.mark.parametrize("channels", [512, 1232, 4096])
@pytest.mark.parametrize("dtype", get_dtypes())
def test_bias_relu(batch, sequence, channels, dtype):
    activations_ds = torch.randn((batch, sequence, channels), dtype=dtype, device=get_accelerator().device_name())
    bias_ds = torch.randn((channels), dtype=dtype, device=get_accelerator().device_name())

    activations_ref = activations_ds.clone().detach()
    bias_ref = bias_ds.clone().detach()

    ds_out = run_bias_relu_ds(activations_ds, bias_ds)
    ref_out = run_bias_relu_reference(activations_ref, bias_ref)
    assert (allclose(ds_out, ref_out))
