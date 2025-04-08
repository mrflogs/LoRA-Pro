# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import torch

from deepspeed.accelerator import get_accelerator
from deepspeed.inference.v2.modules import ConfigBundle
from deepspeed.inference.v2.modules.configs import DSNormConfig
from deepspeed.inference.v2.modules.interfaces import DSPostNormRegistry
from ...v2.inference_test_utils import get_dtypes, allclose


def reference_implementation(residual: torch.Tensor, hidden_states: torch.Tensor, gamma: torch.Tensor,
                             beta: torch.Tensor, epsilon: float) -> torch.Tensor:
    residual_f = residual.to(torch.float32)
    hidden_states_f = hidden_states.to(torch.float32)
    gamma_f = gamma.to(torch.float32)
    beta_f = beta.to(torch.float32)
    return torch.nn.functional.layer_norm(residual_f + hidden_states_f, (hidden_states_f.size(-1), ),
                                          weight=gamma_f,
                                          bias=beta_f,
                                          eps=epsilon).to(hidden_states.dtype)


@pytest.mark.inference_v2_ops
@pytest.mark.parametrize("tokens, channels", [(1, 2048), (37, 8192), (1280, 768), (2048, 5120)])
@pytest.mark.parametrize("dtype", get_dtypes())
def test_cuda_post_ln_module(tokens: int, channels: int, dtype: torch.dtype) -> None:
    config = DSNormConfig(max_tokens=2048,
                          type="layer_norm",
                          channels=channels,
                          residual_dtype=dtype,
                          input_dtype=dtype,
                          output_dtype=dtype,
                          eps=1e-5)
    bundle = ConfigBundle(name='cuda_post_ln', config=config)

    # Input vals
    hidden_states = torch.randn((tokens, channels), dtype=dtype, device=get_accelerator().current_device_name())
    residual = torch.randn((tokens, channels), dtype=dtype, device=get_accelerator().current_device_name())
    gamma = torch.randn((channels), dtype=torch.float32, device=get_accelerator().current_device_name())
    beta = torch.rand((channels), dtype=torch.float32, device=get_accelerator().current_device_name())
    epsilon = 1e-5

    # Reference output
    ref_output = reference_implementation(residual, hidden_states, gamma, beta, epsilon)

    # New output
    post_ln_module = DSPostNormRegistry.instantiate_config(bundle)
    gamma = post_ln_module.transform_param(gamma)
    beta = post_ln_module.transform_param(beta)
    ds_output, _ = post_ln_module(residual, hidden_states, gamma, beta)

    # Check
    assert allclose(ds_output, ref_output)
