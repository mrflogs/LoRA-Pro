# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
from deepspeed import comm as dist
from torch import nn
from torch.nn import functional as F

from torch.nn.parameter import Parameter
from deepspeed.accelerator import get_accelerator
from deepspeed.module_inject.tp_shard import get_shard_size, get_shard_size_list


class TensorParallelConv2d(nn.Module):

    def __init__(self, conv, rank, world_size, shard_by_oc):
        super().__init__()
        self.rank = rank
        self.world_size = world_size
        self.shard_by_oc = shard_by_oc
        self.shard_weights(conv)

    # Split along the input/output channel depending on whether it is the last conv layer.
    def shard_weights(self, conv):
        if self.shard_by_oc:
            total_size = conv.weight.shape[0]
        else:
            total_size = conv.weight.shape[1]
        bias_data = None
        cols_per_rank = [0]
        for i in range(self.world_size - 1, -1, -1):
            cols = total_size // self.world_size
            if i < total_size % self.world_size:
                cols += 1
            cols_per_rank.append(cols_per_rank[-1] + cols)
        weight_data = conv.weight.data
        if self.shard_by_oc:
            # not last conv layer, split output channel
            weight_data = weight_data[cols_per_rank[self.rank]:cols_per_rank[self.rank + 1]]
            if conv.bias is not None:
                bias_data = conv.bias.data[cols_per_rank[self.rank]:cols_per_rank[self.rank + 1]]
        else:
            # last conv layer, split input channel
            weight_data = weight_data[:, cols_per_rank[self.rank]:cols_per_rank[self.rank + 1]]
            if conv.bias is not None:
                bias_data = conv.bias.data / float(self.world_size)
        self.conv = nn.Conv2d(weight_data.shape[1], weight_data.shape[0], conv.kernel_size, conv.stride, conv.padding,
                              conv.dilation, conv.groups, conv.bias is not None, conv.padding_mode)
        self.conv.weight = torch.nn.Parameter(weight_data)
        if conv.bias is not None:
            self.conv.bias = torch.nn.Parameter(bias_data)
        del conv

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.conv(input)


class TensorParallelOcShardConv2d(TensorParallelConv2d):

    def __init__(self, conv, rank, world_size):
        super().__init__(conv, rank, world_size, True)


class TensorParallelIcShardConv2d(TensorParallelConv2d):

    def __init__(self, conv, rank, world_size):
        super().__init__(conv, rank, world_size, False)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out = self.conv(input)
        if self.world_size > 1:
            dist.inference_all_reduce(out)
        return out


class LinearAllreduce(nn.Module):

    def __init__(self, weight, bias=None, mp_group=None):
        super(LinearAllreduce, self).__init__()
        self.weight = weight
        self.bias = bias
        self.mp_group = mp_group

    def forward(self, input):
        output = torch.matmul(input, self.weight.transpose(-1, -2))
        if self.mp_group is not None:
            dist.inference_all_reduce(output, group=self.mp_group)
        if self.bias is not None:
            output += self.bias
        return output


class LmHeadLinearAllreduce(nn.Module):

    def __init__(
        self,
        weight,
        rank,
        world_size,
        bias=None,
        mp_group=None,
    ):
        super(LmHeadLinearAllreduce, self).__init__()
        self.weight = weight
        self.bias = bias
        self.mp_group = mp_group
        self.rank = rank
        self.world_size = world_size

    def forward(self, input):
        input_shard_size = get_shard_size(input.shape[-1], self.world_size, "lm_head")
        input_shard_offset = sum(get_shard_size_list(input.shape[-1], self.world_size, "lm_head")[0:self.rank])
        output = torch.matmul(input[:, :, input_shard_offset:input_shard_offset + input_shard_size],
                              self.weight.transpose(-1, -2))
        if self.mp_group is not None:
            dist.inference_all_reduce(output, group=self.mp_group)
        if self.bias is not None:
            output += self.bias
        return output


class LinearLayer(nn.Module):

    def __init__(self, weight_shape=None, dtype=torch.half, weight=None, bias=None):
        super(LinearLayer, self).__init__()
        if weight is not None:
            self.weight = weight
            self.bias = bias
        else:
            self.weight = Parameter(
                torch.empty(weight_shape, dtype=dtype, device=get_accelerator().current_device_name()))

            self.bias = Parameter(
                torch.empty(weight_shape[0],
                            dtype=dtype,
                            device=get_accelerator().current_device_name())) \
                if bias is not None else None

    def forward(self, input):
        output = torch.matmul(input, self.weight.transpose(-1, -2))
        if self.bias is not None:
            output += self.bias
        return output


class Normalize(nn.Module):

    def __init__(self, dim=None, dtype=torch.float, eps=1e-5, weight=None, bias=None):
        super(Normalize, self).__init__()
        if weight is not None:
            self.weight = weight
            self.bias = bias
        else:
            self.norm = nn.LayerNorm(dim, eps=eps).to(dtype).to(get_accelerator().current_device_name())
            self.weight = self.norm.weight
            self.bias = self.norm.bias

        self.eps = eps

    def forward(self, input):
        return nn.functional.layer_norm(input, input.shape[-1:], self.weight, self.bias, eps=self.eps)


class EmbeddingLayer(nn.Module):

    def __init__(self, weight_shape=None, dtype=torch.half, weight=None, bias=None):
        super(EmbeddingLayer, self).__init__()
        if weight is None:
            self.weight = Parameter(
                torch.empty(weight_shape[0],
                            weight_shape[1],
                            dtype=dtype,
                            device=get_accelerator().current_device_name()))
        else:
            self.weight = weight

    def forward(self, input):
        return F.embedding(input, self.weight)


class OPTEmbedding(EmbeddingLayer):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def __init__(self, weight_shape=None, weight=None, bias=None):
        # OPT is set up so that if padding_idx is specified then offset the embedding ids by 2
        # and adjust num_embeddings appropriately. Other models don't have this hack
        self.offset = 2
        super().__init__(weight_shape, weight=weight)

    def forward(self, attention_mask: torch.LongTensor, past_key_values_length: int = 0):
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        attention_mask = attention_mask.long()

        # create positions depending on attention_mask
        positions = (torch.cumsum(attention_mask, dim=1).type_as(attention_mask) * attention_mask).long() - 1

        # cut positions if `past_key_values_length` is > 0
        positions = positions[:, past_key_values_length:]

        return super().forward(positions + self.offset)


class RMSNormalize(nn.Module):

    def __init__(self, dim=None, dtype=torch.float, eps=1e-5, weight=None):
        super(RMSNormalize, self).__init__()
        if weight is not None:
            self.weight = weight
        else:
            self.weight = nn.Parameter(torch.ones(dim, dtype=dtype, device=get_accelerator().current_device_name()))

        self.eps = eps

    def forward(self, hidden_states):
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return hidden_states * self.weight
