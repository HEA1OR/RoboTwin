#!/usr/bin/env python3
"""Compare pi05 params vs per checkpoint params on VLM + action-expert only."""

from __future__ import annotations

import argparse
import re

import numpy as np
from flax import traverse_util

from openpi.models import model as model_lib


def _is_expert2_llm_key(key: str) -> bool:
    # Matches path segments like ".../xxx_2/..."
    return re.search(r"(^|/)[^/]+_2(/|$)", key) is not None


def _include_vlm_action_only(key: str) -> bool:
    # Exclude perception/critic branches.
    if key.startswith("critic_") or "/critic_" in key:
        return False
    if key.startswith("perception_") or "/perception_" in key:
        return False
    if key.startswith("cond_") or "/cond_" in key:
        return False

    # VLM vision tower.
    if key.startswith("PaliGemma/img/"):
        return True

    # VLM + action expert in shared LLM, excluding expert_2 (perception expert).
    if key.startswith("PaliGemma/llm/"):
        return not _is_expert2_llm_key(key)

    # Action projection heads.
    if key.startswith("action_"):
        return True

    return False


def _to_f32(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float32)


def _to_bf16_then_f32(x: np.ndarray) -> np.ndarray:
    """Cast tensor to bf16 then back to f32 for fair comparison with frozen bf16 checkpoints."""
    try:
        bf16_dtype = np.dtype("bfloat16")
        return np.asarray(np.asarray(x, dtype=bf16_dtype), dtype=np.float32)
    except TypeError:
        import ml_dtypes

        return np.asarray(np.asarray(x, dtype=ml_dtypes.bfloat16), dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pi05-params",
        default="/home/ps/.cache/openpi/openpi-assets/checkpoints/pi05_base/params",
        help="Path to pi05 base params checkpoint dir.",
    )
    parser.add_argument(
        "--target-params",
        default="/home/ps/xwj/RoboTwin/policy/per/checkpoints/per_critic_aloha_robotwin/cb_per_4090/200/params",
        help="Path to target params checkpoint dir.",
    )
    parser.add_argument(
        "--pi05-ref-type",
        choices=("bf16", "fp32"),
        default="bf16",
        help="Reference casting for pi05 params before comparison (default: bf16).",
    )
    parser.add_argument("--tol", type=float, default=1e-3, help="Max-abs threshold for 'changed'.")
    parser.add_argument("--topk", type=int, default=30, help="Print top-K changed keys.")
    args = parser.parse_args()

    pi05 = model_lib.restore_params(args.pi05_params, restore_type=np.ndarray)
    target = model_lib.restore_params(args.target_params, restore_type=np.ndarray)

    flat_pi05 = traverse_util.flatten_dict(pi05, sep="/")
    flat_target = traverse_util.flatten_dict(target, sep="/")

    selected = sorted(k for k in flat_pi05 if _include_vlm_action_only(k))
    missing = [k for k in selected if k not in flat_target]

    shape_mismatch: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = []
    changed: list[tuple[str, float, float]] = []
    unchanged = 0

    for k in selected:
        if k not in flat_target:
            continue
        a = np.asarray(flat_pi05[k])
        b = np.asarray(flat_target[k])
        if a.shape != b.shape:
            shape_mismatch.append((k, a.shape, b.shape))
            continue
        if args.pi05_ref_type == "bf16":
            da = _to_bf16_then_f32(a)
        else:
            da = _to_f32(a)
        db = _to_f32(b)
        diff = np.abs(da - db)
        max_abs = float(np.max(diff))
        mean_abs = float(np.mean(diff))
        if max_abs > args.tol:
            changed.append((k, max_abs, mean_abs))
        else:
            unchanged += 1

    print("==== Compare VLM + Action Expert Params ====")
    print(f"pi05 params:   {args.pi05_params}")
    print(f"pi05 ref type: {args.pi05_ref_type}")
    print(f"target params: {args.target_params}")
    print(f"selected keys from pi05: {len(selected)}")
    print(f"missing in target:       {len(missing)}")
    print(f"shape mismatch:          {len(shape_mismatch)}")
    print(f"changed (max_abs>{args.tol}): {len(changed)}")
    print(f"unchanged (<=tol):       {unchanged}")

    if missing:
        print("\n-- Missing keys (first 20) --")
        for k in missing[:20]:
            print(k)

    if shape_mismatch:
        print("\n-- Shape mismatches (first 20) --")
        for k, sa, sb in shape_mismatch[:20]:
            print(f"{k}: pi05={sa}, target={sb}")

    if changed:
        changed.sort(key=lambda x: x[1], reverse=True)
        print(f"\n-- Top {min(args.topk, len(changed))} changed keys by max_abs --")
        for k, mx, mn in changed[: args.topk]:
            print(f"{k}: max_abs={mx:.6e}, mean_abs={mn:.6e}")


if __name__ == "__main__":
    main()
