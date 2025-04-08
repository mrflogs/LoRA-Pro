# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from typing import Any

from ...config_v2 import RaggedInferenceEngineConfig
from ..inference_policy_base import ContainerMap, InferenceV2Policy
from .container import FalconNonTransformerContainer, FalconTransformerContainer
from .container import FalconNewArchTransformerContainer
from .model import FalconInferenceModel


class FalconPolicy(InferenceV2Policy):

    def instantiate_model(self, engine_config: RaggedInferenceEngineConfig, mp_group: Any) -> FalconInferenceModel:
        return FalconInferenceModel(config=self._model_config, engine_config=engine_config, base_mp_group=mp_group)

    def build_container_map(self) -> ContainerMap:
        map = ContainerMap()

        trans_container_cls = FalconNewArchTransformerContainer if self._model_config.new_decoder_architecture else FalconTransformerContainer
        transformer_containers = [trans_container_cls(self.model) for _ in range(self.model.num_layers)]

        map.set_transformer_params(['transformer.h'], transformer_containers)

        map.set_non_transformer_params(FalconNonTransformerContainer(self.model))

        map.set_unmapped_params(
            [f'model.layers.{i}.self_attn.rotary_emb.inv_freq' for i in range(self.model.num_layers)])

        return map
