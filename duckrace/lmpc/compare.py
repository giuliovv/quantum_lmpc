from __future__ import annotations

from dataclasses import dataclass
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
        augment_cfg=LMPCAugmentConfig(enabled=True),
    )

    return baseline, quantum

