import dataclasses
from typing import TYPE_CHECKING

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma as _gemma
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

if TYPE_CHECKING:
    from openpi.models.pi0 import Pi0


@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"
    perception_expert_variant: _gemma.Variant = "gemma_300m"

    # Set the model specific defaults.
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = None  # type: ignore
    # Pi05 has two differences from Pi0:
    # - the state input is part of the discrete language tokens rather than a continuous input that is part of the suffix
    # - the action expert uses adaRMSNorm to inject the flow matching timestep
    pi05: bool = False
    # This config option is not used directly by the model, but it is read by the ModelTransformFactory.
    discrete_state_input: bool = None  # type: ignore
    # If true, append a second diffusion expert (perception expert) after action expert.
    use_perception_expert: bool = False
    # Number of condition tokens extracted from action expert output and fed to perception expert.
    num_cond_tokens: int = 5
    # Number of perception targets predicted by perception expert.
    perception_horizon: int = 5
    # 1: action-only pretrain; 2: action+perception (GT cond); 3: action+perception (action cond)
    training_stage: int = 1
    # Weights for action/perception diffusion losses.
    action_loss_weight: float = 1.0
    perception_loss_weight: float = 1.0
    # If true, instantiate a critic head over pooled VLM embeddings.
    use_critic: bool = False

    def __post_init__(self):
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 200 if self.pi05 else 48)
        if self.discrete_state_input is None:
            object.__setattr__(self, "discrete_state_input", self.pi05)
        if self.num_cond_tokens < 1 or self.num_cond_tokens > 10:
            raise ValueError(f"num_cond_tokens must be in [1, 10], got {self.num_cond_tokens}")
        if self.perception_horizon < 1 or self.perception_horizon > 10:
            raise ValueError(f"perception_horizon must be in [1, 10], got {self.perception_horizon}")
        if self.training_stage not in (1, 2, 3):
            raise ValueError(f"training_stage must be one of 1/2/3, got {self.training_stage}")

    @property
    @override
    def model_type(self) -> _model.ModelType:
        if self.pi05:
            return _model.ModelType.PI05
        return _model.ModelType.PI0

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0":
        from openpi.models.pi0 import Pi0

        return Pi0(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
                left_arm_state=jax.ShapeDtypeStruct([batch_size], jnp.float32),
                right_arm_state=jax.ShapeDtypeStruct([batch_size], jnp.float32),
                left_arm_dim=jax.ShapeDtypeStruct([batch_size], jnp.float32),
                right_arm_dim=jax.ShapeDtypeStruct([batch_size], jnp.float32),
                perception_arm_dim=jax.ShapeDtypeStruct([batch_size], jnp.float32),
                perception_qpos=jax.ShapeDtypeStruct([batch_size, 5], jnp.float32),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        filters = []
        has_lora = False
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        perception_expert_params_filter = nnx_utils.PathRegex(".*llm.*_2.*")
        if "lora" in self.paligemma_variant:
            filters.append(
                gemma_params_filter,
            )
            if "lora" not in self.action_expert_variant:
                # If only freeze gemma params, exclude action expert params.
                filters.append(
                    nnx.Not(action_expert_params_filter),
                )
            if self.use_perception_expert and "lora" not in self.perception_expert_variant:
                filters.append(
                    nnx.Not(perception_expert_params_filter),
                )
            has_lora = True
        if "lora" in self.action_expert_variant:
            filters.append(
                action_expert_params_filter,
            )
            has_lora = True
        if self.use_perception_expert and "lora" in self.perception_expert_variant:
            filters.append(
                perception_expert_params_filter,
            )
            has_lora = True

        if has_lora:
            # If any lora is used, exclude all lora params.
            filters.append(
                nnx.Not(nnx_utils.PathRegex(".*lora.*")),
            )
        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)
