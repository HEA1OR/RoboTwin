import argparse
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _plot_one(h5_path: Path) -> bool:
    try:
        with h5py.File(h5_path, "r") as f:
            if "/frame_tracking/observer_in_operator_spherical" not in f:
                print(f"[skip] missing observer_in_operator_spherical: {h5_path}")
                return False
            if "/frame_tracking/observer_dir_offset_angles" not in f:
                print(f"[skip] missing observer_dir_offset_angles: {h5_path}")
                return False
            sph = np.array(f["/frame_tracking/observer_in_operator_spherical"])
            off = np.array(f["/frame_tracking/observer_dir_offset_angles"])
            obs_arm = np.array(f["/frame_tracking/observer_arm"]) if "/frame_tracking/observer_arm" in f else None
    except Exception as e:
        print(f"[skip] open/read failed: {h5_path} ({e})")
        return False

    if obs_arm is not None:
        try:
            decoded = np.array([x.decode("utf-8") for x in obs_arm])
            if np.all(decoded == "none"):
                print(f"[skip] observer_arm all none: {h5_path}")
                return False
        except Exception:
            pass

    out_dir = h5_path.parent
    stem = h5_path.stem

    # 1) spherical all dims: r, azimuth_deg, polar_deg
    fig, ax = plt.subplots(figsize=(10, 4.2), dpi=160)
    ax.plot(sph[:, 0], label="r")
    ax.plot(sph[:, 1], label="azimuth_deg")
    ax.plot(sph[:, 2], label="polar_deg")
    ax.set_title("observer_in_operator_spherical")
    ax.set_xlabel("frame")
    ax.set_ylabel("value")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"{stem}_observer_in_operator_spherical.png")
    plt.close(fig)

    # 2) offset angles
    fig, ax = plt.subplots(figsize=(10, 4.2), dpi=160)
    ax.plot(off[:, 0], label="offset_azimuth_deg")
    ax.plot(off[:, 1], label="offset_polar_deg")
    ax.set_title("observer_dir_offset_angles")
    ax.set_xlabel("frame")
    ax.set_ylabel("degree")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"{stem}_observer_dir_offset_angles.png")
    plt.close(fig)

    # 3) radius
    fig, ax = plt.subplots(figsize=(10, 4.2), dpi=160)
    ax.plot(sph[:, 0], label="r")
    ax.set_title("observer_spherical_radius")
    ax.set_xlabel("frame")
    ax.set_ylabel("meter")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"{stem}_observer_spherical_radius.png")
    plt.close(fig)

    # 4) direction
    fig, ax = plt.subplots(figsize=(10, 4.2), dpi=160)
    ax.plot(sph[:, 1], label="azimuth_deg")
    ax.plot(sph[:, 2], label="polar_deg")
    ax.set_title("observer_spherical_direction")
    ax.set_xlabel("frame")
    ax.set_ylabel("degree")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"{stem}_observer_spherical_direction.png")
    plt.close(fig)

    # 5) observer arm state: left=-1, none=0, right=1
    if obs_arm is not None:
        try:
            decoded = np.array(
                [x.decode("utf-8") if isinstance(x, (bytes, np.bytes_)) else str(x) for x in obs_arm],
                dtype=object,
            )
            state = np.zeros(decoded.shape[0], dtype=np.int8)
            state[decoded == "left"] = -1
            state[decoded == "right"] = 1

            fig, ax = plt.subplots(figsize=(10, 3.8), dpi=160)
            ax.plot(state, linewidth=1.2, color="#1f77b4", label="observer_arm_state")
            ax.set_title("observer_arm_state (left=-1, none=0, right=1)")
            ax.set_xlabel("frame")
            ax.set_ylabel("state")
            ax.set_yticks([-1, 0, 1])
            ax.set_yticklabels(["left", "none", "right"])
            ax.set_ylim(-1.2, 1.2)
            ax.grid(alpha=0.25)
            ax.legend()
            fig.tight_layout()
            fig.savefig(out_dir / f"{stem}_observer_arm_state.png")
            plt.close(fig)
        except Exception as e:
            print(f"[warn] observer_arm_state plot failed: {h5_path} ({e})")

    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=str,
        default="/home/ps/xwj/RoboTwin/data",
        help="Root data directory to scan.",
    )
    args = parser.parse_args()

    root = Path(args.root)
    files = []
    files.extend(sorted(root.glob("*/*/data_with_observer/*.hdf5")))
    files.extend(sorted(root.glob("*/*/data/*.hdf5")))
    # de-duplicate while preserving order
    seen = set()
    dedup = []
    for p in files:
        if p in seen:
            continue
        seen.add(p)
        dedup.append(p)
    files = dedup
    print(f"found {len(files)} candidate hdf5 files under {root}")

    ok = 0
    for h5_path in files:
        if _plot_one(h5_path):
            ok += 1
            print(f"[ok] {h5_path}")
    print(f"done: generated figures for {ok}/{len(files)} files")


if __name__ == "__main__":
    main()
