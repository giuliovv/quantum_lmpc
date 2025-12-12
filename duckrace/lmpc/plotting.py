from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class TrackGeometry:
    traj_xytheta: np.ndarray  # (T,3)
    inside_xy: np.ndarray  # (T,2)
    outside_xy: np.ndarray  # (T,2)


def track_violation_report(
    *,
    loop_xy: np.ndarray,
    geom: TrackGeometry,
    feasibility_tol: float = 1.05,
) -> dict:
    """
    Checks whether the executed loop stays within the (inside,outside) corridor.

    We report two metrics:
    - distance-to-centerline vs half-width (coarse)
    - cross-section projection t in [0,1] where p = inside + t*(outside-inside) (stricter)

    Returns a dict with:
    - max_dist_to_center
    - max_allowed_dist
    - n_violations
    - frac_violations
    """
    from scipy import spatial

    xy = np.asarray(loop_xy, dtype=float).reshape(-1, 2)
    traj_xy = np.asarray(geom.traj_xytheta[:, :2], dtype=float)
    inside = np.asarray(geom.inside_xy, dtype=float)
    outside = np.asarray(geom.outside_xy, dtype=float)

    kd = spatial.KDTree(traj_xy)
    _, idxs = kd.query(xy, workers=-1)
    idxs = np.asarray(idxs, dtype=int).reshape(-1)

    centers = traj_xy[idxs]
    seg = outside[idxs] - inside[idxs]
    seg_len2 = np.sum(seg * seg, axis=1)
    seg_len2 = np.where(seg_len2 <= 1e-12, 1e-12, seg_len2)
    t = np.sum((xy - inside[idxs]) * seg, axis=1) / seg_len2

    widths = 0.5 * np.linalg.norm(seg, axis=1)
    dists = np.linalg.norm(xy - centers, axis=1)
    allowed = widths * float(feasibility_tol)

    violations = dists > allowed
    proj_violations = (t < 0.0) | (t > 1.0)
    return {
        "max_dist_to_center": float(np.max(dists)) if dists.size else float("nan"),
        "max_allowed_dist": float(np.max(allowed)) if allowed.size else float("nan"),
        "n_violations": int(np.sum(violations)),
        "frac_violations": float(np.mean(violations)) if violations.size else float("nan"),
        "n_proj_violations": int(np.sum(proj_violations)),
        "frac_proj_violations": float(np.mean(proj_violations)) if proj_violations.size else float("nan"),
    }


def plot_laps(
    *,
    out_path: Path,
    geom: TrackGeometry,
    baseline_xy: np.ndarray,
    quantum_xy: Optional[np.ndarray] = None,
    title: str = "LMPC Laps",
) -> None:
    import matplotlib.pyplot as plt

    traj = np.asarray(geom.traj_xytheta, dtype=float)
    inside = np.asarray(geom.inside_xy, dtype=float)
    outside = np.asarray(geom.outside_xy, dtype=float)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(traj[:, 0], traj[:, 1], "k--", lw=1, label="centerline")
    ax.plot(inside[:, 0], inside[:, 1], "g--", lw=1, label="inside")
    ax.plot(outside[:, 0], outside[:, 1], "g--", lw=1, label="outside")

    b = np.asarray(baseline_xy, dtype=float)
    ax.plot(b[:, 0], b[:, 1], color="tab:blue", lw=2, label="baseline (best)")

    if quantum_xy is not None:
        q = np.asarray(quantum_xy, dtype=float)
        ax.plot(q[:, 0], q[:, 1], color="tab:orange", lw=2, label="quantum (best)")

    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.25)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
