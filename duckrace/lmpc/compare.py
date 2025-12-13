from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np

from duckrace.lmpc.runner import LMPCAugmentConfig, LMPCRunConfig, LMPCRunResult, run_lmpc_iterations
from duckrace.quantum.lmpc_augment import QuantumLMPCAugmenterConfig, make_lmpc_runner_augmenter


@dataclass(frozen=True)
class LMPCKit:
    M_lmpc: Callable
    F_model: Callable
    traj_xytheta: np.ndarray
    inside_xy: np.ndarray
    outside_xy: np.ndarray
    x_init: np.ndarray
    idx_init: int
    X_log_first_loop: np.ndarray


def compare_lmpc_baseline_vs_quantum(
    *,
    kit: LMPCKit,
    run_cfg: LMPCRunConfig,
    n_iterations: int,
    quantum_cfg: Optional[QuantumLMPCAugmenterConfig] = None,
    enable_augment: bool = True,
) -> Tuple[LMPCRunResult, LMPCRunResult]:
    """
    Convenience wrapper for a baseline LMPC run vs a quantum-augmented LMPC run.
    """
    baseline = run_lmpc_iterations(
        M_lmpc=kit.M_lmpc,
        F_model=kit.F_model,
        traj_xytheta=kit.traj_xytheta,
        inside_xy=kit.inside_xy,
        outside_xy=kit.outside_xy,
        x_init=np.asarray(kit.x_init, dtype=float).reshape(-1),
        idx_init=int(kit.idx_init),
        X_log_first_loop=np.asarray(kit.X_log_first_loop, dtype=float),
        config=run_cfg,
        n_iterations=n_iterations,
    )

    if quantum_cfg is None:
        quantum_cfg = QuantumLMPCAugmenterConfig()

    augmenter = make_lmpc_runner_augmenter(model_F=kit.F_model, cfg=quantum_cfg)
    quantum = run_lmpc_iterations(
        M_lmpc=kit.M_lmpc,
        F_model=kit.F_model,
        traj_xytheta=kit.traj_xytheta,
        inside_xy=kit.inside_xy,
        outside_xy=kit.outside_xy,
        x_init=np.asarray(kit.x_init, dtype=float).reshape(-1),
        idx_init=int(kit.idx_init),
        X_log_first_loop=np.asarray(kit.X_log_first_loop, dtype=float),
        config=run_cfg,
        n_iterations=n_iterations,
        augmenter=augmenter,
        augment_cfg=LMPCAugmentConfig(enabled=enable_augment),
    )

    return baseline, quantum


def lap_times_seconds(result: LMPCRunResult, *, include_seed_lap: bool = True) -> np.ndarray:
    loops = result.plain_loops if include_seed_lap else result.plain_loops[1:]
    return np.asarray([loop.shape[1] * float(result.dt) for loop in loops], dtype=float)


def plot_lap_times(
    *,
    out_path: Path,
    baseline: LMPCRunResult,
    quantum: Optional[LMPCRunResult] = None,
    include_seed_lap: bool = True,
    title: str = "Lap time per iteration",
) -> None:
    import matplotlib.pyplot as plt

    b_times = lap_times_seconds(baseline, include_seed_lap=include_seed_lap)
    x = np.arange(b_times.shape[0])

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x, b_times, marker="o", lw=2, color="tab:blue", label="classical")

    if quantum is not None:
        q_times = lap_times_seconds(quantum, include_seed_lap=include_seed_lap)
        x_q = np.arange(q_times.shape[0])
        ax.plot(x_q, q_times, marker="o", lw=2, color="tab:orange", label="quantum")

    ax.set_title(title)
    ax.set_xlabel("Lap / iteration")
    ax.set_ylabel("Lap time (s)")
    ax.grid(True, alpha=0.25)
    ax.legend()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
