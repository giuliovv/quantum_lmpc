from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .discretization import decode_control_sequence
from .sampler import QuantumControlSampler, QuantumSamplerConfig


@dataclass(frozen=True)
class QuantumLMPCAugmenterConfig:
    horizon: int = 4
    n_start_states: int = 8
    n_samples_per_start: int = 4
    sampler: QuantumSamplerConfig = QuantumSamplerConfig(backend="statevector", n_iterations=1, seed=0)
    feasibility_tol: float = 1.05
    # Only keep samples whose terminal state is near a safe-set point that is at least
    # `min_advance_steps` "steps-to-go" ahead of the expected progress (J_start - horizon).
    # This prevents adding slow/redundant points that don't improve lap time.
    min_advance_steps: int = 1
    # Reject augmented points with very low terminal speed; these tend to degrade LMPC.
    min_terminal_v: float = 0.45
    # Also require terminal v >= quantile of the current lap's v distribution.
    # This helps prevent adding "slow-but-feasible" points that drag the policy down.
    min_terminal_v_quantile: float = 0.5


def _is_on_track(
    xy: np.ndarray,
    idx: int,
    traj_xy: np.ndarray,
    inside_xy: np.ndarray,
    outside_xy: np.ndarray,
    tol: float,
) -> bool:
    center = traj_xy[idx]
    half_width = 0.5 * float(np.linalg.norm(inside_xy[idx] - outside_xy[idx]))
    dist = float(np.linalg.norm(np.asarray(xy, dtype=float) - center))
    return dist <= (half_width * tol)


def augment_safe_set_with_quantum(
    last_points_with_time: np.ndarray,
    traj_xytheta: np.ndarray,
    inside_xy: np.ndarray,
    outside_xy: np.ndarray,
    model_F,
    cfg: QuantumLMPCAugmenterConfig = QuantumLMPCAugmenterConfig(),
) -> np.ndarray:
    """
    Augment a single lap safe set (6,T) by sampling extra terminal states
    that minimize the estimated time-to-go `J` under the current safe set.

    Cost table used for Grover sampling:
      cost(seq) = horizon + J(nearest_safe_state(x_T))

    Feasibility:
      we keep only samples whose terminal (x,y) is within the track corridor
      approximated by distance-to-centerline <= half-width at nearest index.
    """
    from scipy import spatial

    last_points_with_time = np.asarray(last_points_with_time, dtype=float)
    if last_points_with_time.ndim != 2 or last_points_with_time.shape[0] != 6:
        raise ValueError("last_points_with_time must have shape (6, T).")

    horizon = int(cfg.horizon)
    n_total = 4**horizon
    traj_xy = np.asarray(traj_xytheta[:, :2], dtype=float)

    kd_traj = spatial.KDTree(traj_xy)
    kd_safe = spatial.KDTree(last_points_with_time[:2].T)

    safe_J = last_points_with_time[-1].reshape(-1)
    T = last_points_with_time.shape[1]
    start_idxs = np.linspace(0, max(0, T - 2), max(1, min(cfg.n_start_states, T)), dtype=int)
    start_idxs = np.unique(start_idxs)

    sampler = QuantumControlSampler(n_qubits=2 * horizon, config=cfg.sampler)

    new_cols: list[np.ndarray] = []

    for si in start_idxs:
        x0 = last_points_with_time[:-1, si].reshape(-1)
        J_start = float(last_points_with_time[-1, si])  # steps-to-go at start state

        v_floor = float(cfg.min_terminal_v)
        if last_points_with_time.shape[0] > 3:
            lap_v = np.asarray(last_points_with_time[3, :], dtype=float).reshape(-1)
            lap_v = lap_v[np.isfinite(lap_v)]
            if lap_v.size > 0:
                q = float(np.clip(cfg.min_terminal_v_quantile, 0.0, 1.0))
                v_floor = max(v_floor, float(np.quantile(lap_v, q)))

        # Build cost table by rolling out all 4^h sequences and scoring by nearest safe J.
        costs = np.full((n_total,), float("inf"), dtype=float)
        for idx in range(n_total):
            wl, wr = decode_control_sequence(idx, horizon=horizon)
            x = x0.copy()
            for t in range(horizon):
                u = np.array([wl[t], wr[t]], dtype=float)
                x = np.asarray(model_F(x, u)).reshape(-1)
            xy = x[:2]
            vT = float(x[3]) if x.size > 3 else float("nan")
            _, traj_idx = kd_traj.query(xy.reshape(-1), workers=-1)
            if not _is_on_track(xy, int(traj_idx), traj_xy, inside_xy, outside_xy, tol=cfg.feasibility_tol):
                continue
            _, safe_idx = kd_safe.query(xy.reshape(-1), workers=-1)
            safe_idx = int(safe_idx)

            # Filter out clearly slow terminal states (usually from {0,1} wheel discretization).
            # Use an absolute threshold to avoid dependence on v_start (which can be ~0 early on).
            if np.isfinite(vT) and float(vT) < float(v_floor):
                continue

            # Require that terminal xy "snaps" to a strictly later safe-set point than
            # we'd get by just advancing `horizon` steps along the current loop.
            expected_J = J_start - float(horizon)
            advance = expected_J - float(safe_J[safe_idx])
            if advance < float(cfg.min_advance_steps):
                continue

            # Cost is "how much we advanced" (more advance => lower cost).
            costs[idx] = float(-advance)

        # If everything is infeasible, skip.
        finite = costs[np.isfinite(costs)]
        if finite.size == 0:
            continue
        thr = float(np.percentile(finite, 25.0))

        sampled = sampler.sample(cost_table=costs, threshold=thr, n_samples=int(cfg.n_samples_per_start))

        # Track rejection reasons for diagnostics
        n_infeasible = 0
        n_low_v = 0
        n_no_advance = 0
        n_accepted = 0

        for sidx in sampled:
            if not np.isfinite(costs[int(sidx)]):
                n_infeasible += 1
                continue
            wl, wr = decode_control_sequence(int(sidx), horizon=horizon)
            x = x0.copy()
            for t in range(horizon):
                u = np.array([wl[t], wr[t]], dtype=float)
                x = np.asarray(model_F(x, u)).reshape(-1)
            xy = x[:2]
            vT = float(x[3]) if x.size > 3 else float("nan")
            if np.isfinite(vT) and float(vT) < float(v_floor):
                n_low_v += 1
                continue

            _, safe_idx = kd_safe.query(xy.reshape(-1), workers=-1)
            safe_idx = int(safe_idx)
            expected_J = J_start - float(horizon)
            advance = expected_J - float(safe_J[safe_idx])
            if advance < float(cfg.min_advance_steps):
                n_no_advance += 1
                continue

            n_accepted += 1
            # Label J with the snapped safe-set point (progress along the lap).
            J_new = float(safe_J[safe_idx])
            new_cols.append(np.concatenate([x.reshape(-1), np.array([J_new], dtype=float)]))

        # Log rejection stats if verbose (check for environment variable)
        import os
        if os.environ.get("QUANTUM_LMPC_VERBOSE"):
            n_feasible_in_table = int(np.sum(np.isfinite(costs)))
            print(f"  [augment] start={si}: samples={len(sampled)}, feasible_table={n_feasible_in_table}/{n_total}, "
                  f"rejected: infeasible={n_infeasible}, low_v={n_low_v}, no_advance={n_no_advance}, accepted={n_accepted}")

    if not new_cols:
        return last_points_with_time

    extra = np.stack(new_cols, axis=1)  # (6, n_extra)
    return np.hstack([last_points_with_time, extra])


def make_lmpc_runner_augmenter(
    *,
    model_F,
    cfg: QuantumLMPCAugmenterConfig = QuantumLMPCAugmenterConfig(),
):
    """
    Convenience adapter for `duckrace.lmpc.run_lmpc_iterations(..., augmenter=...)`.
    """

    def _augment(last_points_with_time, traj_xytheta, inside_xy, outside_xy, _augment_cfg):
        return augment_safe_set_with_quantum(
            last_points_with_time=last_points_with_time,
            traj_xytheta=traj_xytheta,
            inside_xy=inside_xy,
            outside_xy=outside_xy,
            model_F=model_F,
            cfg=cfg,
        )

    return _augment
