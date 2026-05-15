import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


def _interp_one_batch(actions: at.Float[at.Array, "s d"], source_t, target_t):
    """Linear interpolation of one action sequence in time."""

    def interp_dim(x: at.Float[at.Array, "s"]) -> at.Float[at.Array, "t"]:
        return jnp.interp(target_t, source_t, x)

    return jax.vmap(interp_dim, in_axes=1, out_axes=1)(actions)


class Pi0(_model.BaseModel):
    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        self.use_perception_expert = config.use_perception_expert
        self.use_critic = config.use_critic
        self.num_cond_tokens = config.num_cond_tokens
        self.perception_horizon = config.perception_horizon
        self.training_stage = config.training_stage
        self.action_loss_weight = config.action_loss_weight
        self.perception_loss_weight = config.perception_loss_weight

        if self.use_perception_expert and not self.pi05:
            raise ValueError("Perception expert is currently supported only when pi05=True.")

        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)

        llm_configs = [paligemma_config, action_expert_config]
        use_adarms = [False, True] if config.pi05 else [False, False]

        if self.use_perception_expert:
            perception_expert_config = _gemma.get_config(config.perception_expert_variant)
            llm_configs.append(perception_expert_config)
            use_adarms.append(config.pi05)
        else:
            perception_expert_config = None
            self.perception_horizon = 0

        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=llm_configs,
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=use_adarms)
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)

        # Action expert projections.
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        if config.pi05:
            self.action_time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        # Perception expert projections and cond structure.
        if self.use_perception_expert:
            assert perception_expert_config is not None
            self.cond_mlp_in = nnx.Linear(config.action_dim, perception_expert_config.width, rngs=rngs)
            self.cond_mlp_out = nnx.Linear(perception_expert_config.width, perception_expert_config.width, rngs=rngs)
            self.cond_end_proj = nnx.Linear(1, perception_expert_config.width, rngs=rngs)
            self.perception_in_proj = nnx.Linear(config.action_dim, perception_expert_config.width, rngs=rngs)
            self.perception_out_proj = nnx.Linear(perception_expert_config.width, config.action_dim, rngs=rngs)
            if config.pi05:
                self.perception_time_mlp_in = nnx.Linear(
                    perception_expert_config.width, perception_expert_config.width, rngs=rngs
                )
                self.perception_time_mlp_out = nnx.Linear(
                    perception_expert_config.width, perception_expert_config.width, rngs=rngs
                )

        # Optional critic head over pooled VLM prefix embeddings.
        if self.use_critic:
            critic_hidden = paligemma_config.width
            self.critic_in = nnx.Linear(paligemma_config.width, critic_hidden, rngs=rngs)
            self.critic_mid = nnx.Linear(critic_hidden, critic_hidden // 2, rngs=rngs)
            self.critic_out = nnx.Linear(critic_hidden // 2, 2, rngs=rngs)

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True

    def _expert_inputs(self, prefix_tokens, action_tokens, perception_tokens=None):
        if self.use_perception_expert:
            return [prefix_tokens, action_tokens, perception_tokens]
        return [prefix_tokens, action_tokens]

    def _adarms_inputs(self, action_cond=None, perception_cond=None):
        if self.use_perception_expert:
            return [None, action_cond, perception_cond]
        return [None, action_cond]

    @staticmethod
    def _sample_diffusion_time(rng: at.KeyArrayLike, batch_shape):
        return jax.random.beta(rng, 1.5, 1, batch_shape) * 0.999 + 0.001

    @staticmethod
    def _make_noisy_actions(
        noise_rng: at.KeyArrayLike,
        time: at.Float[at.Array, " b"],
        actions: _model.Actions,
    ) -> tuple[_model.Actions, _model.Actions]:
        noise = jax.random.normal(noise_rng, actions.shape)
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions
        return x_t, u_t

    def _time_embedding(
        self,
        timestep: at.Float[at.Array, " b"],
        out_dim: int,
        mlp_in: nnx.Linear,
        mlp_out: nnx.Linear,
    ) -> at.Float[at.Array, "b emb"]:
        time_emb = posemb_sincos(timestep, out_dim, min_period=4e-3, max_period=4.0)
        time_emb = mlp_in(time_emb)
        time_emb = nnx.swish(time_emb)
        time_emb = mlp_out(time_emb)
        time_emb = nnx.swish(time_emb)
        return time_emb

    @staticmethod
    def _uniform_indices(seq_len: int, target_len: int) -> at.Int[at.Array, " n"]:
        if target_len == 1:
            return jnp.zeros((1,), dtype=jnp.int32)
        return jnp.round(jnp.linspace(0, seq_len - 1, target_len)).astype(jnp.int32)

    def _subsample_actions(
        self,
        actions: at.Float[at.Array, "b h d"],
        target_len: int,
    ) -> at.Float[at.Array, "b n d"]:
        idx = self._uniform_indices(actions.shape[1], target_len)
        return jnp.take(actions, idx, axis=1)

    def _interpolate_actions(
        self,
        actions: at.Float[at.Array, "b s d"],
        target_len: int,
    ) -> at.Float[at.Array, "b t d"]:
        source_len = actions.shape[1]
        if source_len == target_len:
            return actions
        source_t = jnp.linspace(0.0, 1.0, source_len, dtype=actions.dtype)
        target_t = jnp.linspace(0.0, 1.0, target_len, dtype=actions.dtype)
        return jax.vmap(lambda x: _interp_one_batch(x, source_t, target_t))(actions)

    def _state_dim_masks(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b d"], at.Float[at.Array, "b d"]]:
        """Build per-dimension masks for operate-action loss and observe-perception loss."""
        batch_size = obs.state.shape[0]

        # Action layout in padded 32-D vector:
        #   left action dims  -> [0, left_arm_dim + 1)   (+1 for gripper)
        #   right action dims -> [left_arm_dim + 1, left_arm_dim + 1 + right_arm_dim + 1)
        left_action_dim = (
            jnp.ravel(jnp.asarray(obs.left_arm_dim).astype(jnp.int32)) + 1
            if obs.left_arm_dim is not None
            else jnp.full((batch_size,), 7, dtype=jnp.int32)
        )
        left_action_dim = jnp.clip(left_action_dim, 0, self.action_dim)
        right_action_dim = (
            jnp.ravel(jnp.asarray(obs.right_arm_dim).astype(jnp.int32)) + 1
            if obs.right_arm_dim is not None
            else jnp.full((batch_size,), 7, dtype=jnp.int32)
        )
        right_action_dim = jnp.clip(right_action_dim, 0, self.action_dim - left_action_dim)

        # Perception layout in padded 32-D vector:
        #   left observe dims  -> [0, perception_arm_dim)
        #   right observe dims -> [perception_arm_dim, 2 * perception_arm_dim)
        # `perception_arm_dim` is the per-arm observation feature dimension.
        per_arm_dim = (
            jnp.ravel(jnp.asarray(obs.perception_arm_dim).astype(jnp.int32))
            if obs.perception_arm_dim is not None
            else jnp.full((batch_size,), 5, dtype=jnp.int32)
        )
        per_arm_dim = jnp.clip(per_arm_dim, 0, self.action_dim // 2)

        dim_idx = jnp.arange(self.action_dim)[None, :]
        action_left_mask = dim_idx < left_action_dim[:, None]
        action_right_start = left_action_dim[:, None]
        action_right_end = action_right_start + right_action_dim[:, None]
        action_right_mask = jnp.logical_and(dim_idx >= action_right_start, dim_idx < action_right_end)
        perception_left_mask = dim_idx < per_arm_dim[:, None]
        perception_right_start = per_arm_dim[:, None]
        perception_right_end = perception_right_start + per_arm_dim[:, None]
        perception_right_mask = jnp.logical_and(dim_idx >= perception_right_start, dim_idx < perception_right_end)

        if obs.left_arm_state is None or obs.right_arm_state is None:
            left_operate = jnp.ones((batch_size,), dtype=jnp.bool_)
            right_operate = jnp.ones((batch_size,), dtype=jnp.bool_)
        else:
            left_operate = jnp.ravel(jnp.asarray(obs.left_arm_state)) < 0.5
            right_operate = jnp.ravel(jnp.asarray(obs.right_arm_state)) < 0.5
        left_observe = jnp.logical_not(left_operate)
        right_observe = jnp.logical_not(right_operate)

        action_dim_mask = jnp.logical_or(
            jnp.logical_and(action_left_mask, left_operate[:, None]),
            jnp.logical_and(action_right_mask, right_operate[:, None]),
        )
        perception_dim_mask = jnp.logical_or(
            jnp.logical_and(perception_left_mask, left_observe[:, None]),
            jnp.logical_and(perception_right_mask, right_observe[:, None]),
        )
        return action_dim_mask.astype(jnp.float32), perception_dim_mask.astype(jnp.float32)

    @staticmethod
    def _masked_mse(
        pred: at.Float[at.Array, "b h d"],
        target: at.Float[at.Array, "b h d"],
        dim_mask: at.Float[at.Array, "b d"],
        *,
        zero_if_empty: bool,
    ) -> at.Float[at.Array, "b h"]:
        sq = jnp.square(pred - target)
        masked_sq = sq * dim_mask[:, None, :]
        denom = jnp.sum(dim_mask, axis=-1, keepdims=True)
        loss = jnp.sum(masked_sq, axis=-1) / jnp.clip(denom, a_min=1.0)
        if zero_if_empty:
            has_valid = (denom[:, 0] > 0).astype(loss.dtype)
            loss = loss * has_valid[:, None]
        return loss

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        # embed images
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # image tokens attend to each other
            ar_mask += [False] * image_tokens.shape[1]

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # full attention between image and language inputs
            ar_mask += [False] * tokenized_inputs.shape[1]
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_action_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        if not self.pi05:
            # add a single state token
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            # image/language inputs do not attend to state or actions
            ar_mask += [True]

        action_tokens = self.action_in_proj(noisy_actions)
        if self.pi05:
            # time MLP (for adaRMS)
            time_emb = self._time_embedding(
                timestep,
                self.action_in_proj.out_features,
                self.action_time_mlp_in,
                self.action_time_mlp_out,
            )
            action_expert_tokens = action_tokens
            adarms_cond = time_emb
        else:
            # mix timestep + action information using an MLP (no adaRMS)
            time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    @at.typecheck
    def embed_perception_suffix(
        self,
        cond_actions: at.Float[at.Array, "b n d"],
        noisy_perception_actions: at.Float[at.Array, "b n d"],
        timestep: at.Float[at.Array, " b"],
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        if not self.use_perception_expert:
            raise ValueError("Perception expert is disabled in this config.")

        cond_tokens = self.cond_mlp_in(cond_actions)
        cond_tokens = nnx.swish(cond_tokens)
        cond_tokens = self.cond_mlp_out(cond_tokens)
        cond_tokens = nnx.swish(cond_tokens)

        cond_end = self.cond_end_proj(jnp.ones((cond_actions.shape[0], 1, 1), dtype=cond_tokens.dtype))
        perception_tokens = self.perception_in_proj(noisy_perception_actions)

        if self.pi05:
            adarms_cond = self._time_embedding(
                timestep,
                self.perception_in_proj.out_features,
                self.perception_time_mlp_in,
                self.perception_time_mlp_out,
            )
        else:
            adarms_cond = None

        tokens = jnp.concatenate([cond_tokens, cond_end, perception_tokens], axis=1)
        input_mask = jnp.ones(tokens.shape[:2], dtype=jnp.bool_)

        # Structured layout: [cond_1 ... cond_n] [<COND_END>] [target_1 ... target_n]
        # The target block can see all cond tokens and previous shared-context tokens.
        cond_len = cond_tokens.shape[1]
        ar_mask = [False] * cond_len + [True] + [True] + ([False] * (noisy_perception_actions.shape[1] - 1))
        return tokens, input_mask, jnp.array(ar_mask), adarms_cond

    def _run_prefill(
        self,
        prefix_tokens: at.Float[at.Array, "b p d"],
        prefix_mask: at.Bool[at.Array, "b p"],
        prefix_ar_mask: at.Bool[at.Array, " p"],
    ):
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm(
            self._expert_inputs(prefix_tokens, None, None),
            mask=prefix_attn_mask,
            positions=positions,
        )
        return kv_cache

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        if self.use_perception_expert:
            preprocess_rng, noise_rng, time_rng, p_noise_rng, p_time_rng = jax.random.split(rng, 5)
        else:
            preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)

        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        time = self._sample_diffusion_time(time_rng, batch_shape)
        x_t, u_t = self._make_noisy_actions(noise_rng, time, actions)

        # one forward pass of prefix + action suffix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        action_suffix_tokens, action_suffix_mask, action_suffix_ar_mask, action_adarms_cond = self.embed_action_suffix(
            observation, x_t, time
        )

        action_input_mask = jnp.concatenate([prefix_mask, action_suffix_mask], axis=1)
        action_ar_mask = jnp.concatenate([prefix_ar_mask, action_suffix_ar_mask], axis=0)
        action_attn_mask = make_attn_mask(action_input_mask, action_ar_mask)
        action_positions = jnp.cumsum(action_input_mask, axis=1) - 1
        action_outputs, _ = self.PaliGemma.llm(
            self._expert_inputs(prefix_tokens, action_suffix_tokens, None),
            mask=action_attn_mask,
            positions=action_positions,
            adarms_cond=self._adarms_inputs(action_adarms_cond, None),
        )
        action_suffix_out = action_outputs[1]
        v_t_action = self.action_out_proj(action_suffix_out[:, -self.action_horizon :])
        action_dim_mask, perception_dim_mask = self._state_dim_masks(observation)
        action_loss = self._masked_mse(v_t_action, u_t, action_dim_mask, zero_if_empty=True)

        if not self.use_perception_expert or self.training_stage == 1:
            return action_loss

        # Build condition tokens from action expert predictions: x_0_hat = x_t - t * u_hat.
        action_x0_hat = x_t - time[..., None, None] * v_t_action
        if self.training_stage == 2:
            cond_actions = self._subsample_actions(actions, self.num_cond_tokens)
        elif self.training_stage == 3:
            cond_actions = self._subsample_actions(jax.lax.stop_gradient(action_x0_hat), self.num_cond_tokens)
        else:
            raise ValueError(f"Unsupported training_stage={self.training_stage}")
        perception_targets = self._subsample_actions(actions, self.perception_horizon)

        p_time = self._sample_diffusion_time(p_time_rng, batch_shape)
        p_x_t, p_u_t = self._make_noisy_actions(p_noise_rng, p_time, perception_targets)

        perception_suffix_tokens, perception_suffix_mask, perception_suffix_ar_mask, perception_adarms_cond = (
            self.embed_perception_suffix(cond_actions, p_x_t, p_time)
        )

        perception_input_mask = jnp.concatenate([prefix_mask, perception_suffix_mask], axis=1)
        perception_ar_mask = jnp.concatenate([prefix_ar_mask, perception_suffix_ar_mask], axis=0)
        perception_attn_mask = make_attn_mask(perception_input_mask, perception_ar_mask)
        perception_positions = jnp.cumsum(perception_input_mask, axis=1) - 1
        perception_outputs, _ = self.PaliGemma.llm(
            self._expert_inputs(prefix_tokens, None, perception_suffix_tokens),
            mask=perception_attn_mask,
            positions=perception_positions,
            adarms_cond=self._adarms_inputs(None, perception_adarms_cond),
        )
        perception_suffix_out = perception_outputs[2]
        v_t_perception = self.perception_out_proj(perception_suffix_out[:, -self.perception_horizon :])
        perception_masked_loss = self._masked_mse(v_t_perception, p_u_t, perception_dim_mask, zero_if_empty=True)
        perception_mean = jnp.mean(perception_masked_loss, axis=-1, keepdims=True)
        perception_term = jnp.repeat(perception_mean, self.action_horizon, axis=1)
        return self.action_loss_weight * action_loss + self.perception_loss_weight * perception_term

    def _sample_action_expert(
        self,
        observation: _model.Observation,
        prefix_tokens: at.Float[at.Array, "b p d"],
        prefix_mask: at.Bool[at.Array, "b p"],
        kv_cache,
        *,
        num_steps: int,
        noise: at.Float[at.Array, "b ah ad"],
    ) -> _model.Actions:
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_action_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            outputs, _ = self.PaliGemma.llm(
                self._expert_inputs(None, suffix_tokens, None),
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=self._adarms_inputs(adarms_cond, None),
            )
            v_t = self.action_out_proj(outputs[1][:, -self.action_horizon :])
            return x_t + dt * v_t, time + dt

        def cond(carry):
            _, time = carry
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0

    def _sample_perception_expert(
        self,
        prefix_tokens: at.Float[at.Array, "b p d"],
        prefix_mask: at.Bool[at.Array, "b p"],
        kv_cache,
        cond_actions: at.Float[at.Array, "b n d"],
        *,
        num_steps: int,
        noise: at.Float[at.Array, "b n d"],
    ) -> at.Float[at.Array, "b n d"]:
        dt = -1.0 / num_steps
        batch_size = cond_actions.shape[0]

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_perception_suffix(
                cond_actions,
                x_t,
                jnp.broadcast_to(time, batch_size),
            )
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            outputs, _ = self.PaliGemma.llm(
                self._expert_inputs(None, None, suffix_tokens),
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=self._adarms_inputs(None, adarms_cond),
            )
            v_t = self.perception_out_proj(outputs[2][:, -self.perception_horizon :])
            return x_t + dt * v_t, time + dt

        def cond(carry):
            _, time = carry
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        observation = _model.preprocess_observation(None, observation, train=False)
        batch_size = observation.state.shape[0]

        if self.use_perception_expert:
            action_noise_rng, perception_noise_rng = jax.random.split(rng)
        else:
            action_noise_rng = rng
            perception_noise_rng = None

        if noise is None:
            noise = jax.random.normal(action_noise_rng, (batch_size, self.action_horizon, self.action_dim))

        # First fill KV cache with a forward pass of the prefix.
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        kv_cache = self._run_prefill(prefix_tokens, prefix_mask, prefix_ar_mask)

        action_x0 = self._sample_action_expert(
            observation,
            prefix_tokens,
            prefix_mask,
            kv_cache,
            num_steps=num_steps,
            noise=noise,
        )

        if not self.use_perception_expert:
            return action_x0

        cond_actions = self._subsample_actions(action_x0, self.num_cond_tokens)
        perception_noise = jax.random.normal(
            perception_noise_rng,
            (batch_size, self.perception_horizon, self.action_dim),
        )
        perception_x0 = self._sample_perception_expert(
            prefix_tokens,
            prefix_mask,
            kv_cache,
            cond_actions,
            num_steps=num_steps,
            noise=perception_noise,
        )
        return self._interpolate_actions(perception_x0, self.action_horizon)

    def _pool_prefix_output(
        self,
        prefix_out: at.Float[at.Array, "b p d"],
        prefix_mask: at.Bool[at.Array, "b p"],
    ) -> at.Float[at.Array, "b d"]:
        denom = jnp.clip(jnp.sum(prefix_mask, axis=1, keepdims=True), a_min=1)
        return jnp.sum(prefix_out * prefix_mask[..., None], axis=1) / denom

    def critic_scores(
        self,
        observation: _model.Observation,
        *,
        preprocess: bool = True,
    ) -> at.Float[at.Array, "b 2"]:
        if not self.use_critic:
            raise ValueError("Critic is disabled in this config.")

        if preprocess:
            observation = _model.preprocess_observation(None, observation, train=False)

        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        outputs, _ = self.PaliGemma.llm(
            self._expert_inputs(prefix_tokens, None, None),
            mask=prefix_attn_mask,
            positions=positions,
        )
        prefix_out = outputs[0]
        pooled = self._pool_prefix_output(prefix_out, prefix_mask)

        hidden = self.critic_in(pooled)
        hidden = nnx.swish(hidden)
        hidden = self.critic_mid(hidden)
        hidden = nnx.swish(hidden)
        logits = self.critic_out(hidden)
        return jax.nn.sigmoid(logits)

    def compute_critic_loss(
        self,
        observation: _model.Observation,
        *,
        train: bool = False,
    ) -> at.Float[at.Array, "*b"]:
        if observation.left_arm_state is None or observation.right_arm_state is None:
            raise ValueError("left_arm_state and right_arm_state are required for critic loss.")

        scores = self.critic_scores(observation, preprocess=True)
        left_targets = jnp.ravel(jnp.asarray(observation.left_arm_state)).astype(jnp.float32)
        right_targets = jnp.ravel(jnp.asarray(observation.right_arm_state)).astype(jnp.float32)
        targets = jnp.stack([left_targets, right_targets], axis=-1)
        eps = 1e-6
        bce = -(targets * jnp.log(scores + eps) + (1.0 - targets) * jnp.log(1.0 - scores + eps))
        return jnp.mean(bce, axis=-1)
