# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import deepspeed

from deepspeed.utils.torch import required_torch_version
from unit.common import DistributedTest
from unit.simple_model import *
from unit.checkpoint.common import *

import pytest

if not required_torch_version(max_version=2.0):
    pytest.skip("Skipping until we resolve problems with torch 2.1", allow_module_level=True)


class TestMiCSCheckpoint(DistributedTest):
    world_size = 4

    def _toy_model_config(self, shard_size):

        config_dict = {
            "train_micro_batch_size_per_gpu": 2,
            "steps_per_print": 1,
            "optimizer": {
                "type": 'Adam',
                "params": {
                    "lr": 0.00015,
                    "betas": [0.8, 0.999],
                    "eps": 1e-8,
                    "weight_decay": 3e-7
                }
            },
            "fp16": {
                "enabled": True,
                "initial_scale_power": 8
            },
            "wall_clock_breakdown": True,
            "zero_optimization": {
                "stage": 3,
                "mics_shard_size": shard_size
            }
        }

        hidden_dim = 10
        with deepspeed.zero.MiCS_Init(config_dict_or_path=config_dict):
            models = [SimpleModel(hidden_dim, empty_grad=False) for _ in range(2)]

        return config_dict, hidden_dim, models

    @pytest.mark.parametrize('shard_size', [1, 2, 4])
    def test_load_optimizer_state(self, tmpdir, shard_size):
        config_dict, hidden_dim, models = self._toy_model_config(shard_size)
        checkpoint_correctness_verification(config_dict, models, hidden_dim, tmpdir, load_optimizer_states=True)

    @pytest.mark.parametrize('shard_size', [1, 2, 4])
    def test_not_load_optimizer_state(self, tmpdir, shard_size):
        config_dict, hidden_dim, models = self._toy_model_config(shard_size)
        checkpoint_correctness_verification(config_dict, models, hidden_dim, tmpdir, load_optimizer_states=False)

    @pytest.mark.parametrize('shard_size', [1, 2, 4])
    def test_load_module_only(self, tmpdir, shard_size):
        config_dict, hidden_dim, models = self._toy_model_config(shard_size)
        checkpoint_correctness_verification(config_dict, models, hidden_dim, tmpdir, load_module_only=True)

    @pytest.mark.parametrize('shard_size', [1, 2, 4])
    def test_save_checkpoint_on_first_partition_group(self, tmpdir, shard_size):
        config_dict, _, models = self._toy_model_config(shard_size)
        ds_engine, _, _, _ = deepspeed.initialize(config=config_dict,
                                                  model=models[0],
                                                  model_parameters=models[0].parameters(),
                                                  optimizer=None)

        ds_engine.save_checkpoint(tmpdir)
        if ds_engine.global_rank < shard_size:
            assert ds_engine.save_non_zero_checkpoint == True
        else:
            assert ds_engine.save_non_zero_checkpoint == False
