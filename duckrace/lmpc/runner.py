from __future__ import annotations

from dataclasses import dataclass
from timeit import default_timer as timer
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class LMPCRunConfig:
    """
    Minimal configuration to reproduce the LMPC loop in `iterative_mpc_hull_constraint.ipynb`.

    Notes:
    - `N` is the LMPC horizon (same `N` in the notebook, not `N_MPC`).
    - `K` and `i_j` set the number of nearest neighbors K*i_j.
    - `frame_rate` and `finish_t_steps` are in simulator steps.
    """

    N: int
    K: int
    i_j: int
    frame_rate: int
    more: int
    dt: float
    finish_y: float
    finish_x_min: float
    finish_t_steps: int
    max_seconds: float = 60.0
    i_j_all: bool = False
    # If set, we stop when we've advanced ~1 full lap along the centerline index.
    start_traj_index: Optional[int] = None
    traj_len: Optional[int] = None


@dataclass(frozen=True)
class LMPCAugmentConfig:
    enabled: bool = False
    horizon: int = 4
    n_start_states: int = 8
    n_samples_per_start: int = 4
    # The augmenter callable signature is documented below.


@dataclass
class LMPCRunResult:
    loops_with_time: List[np.ndarray]  # each is (6, T) : [x,y,theta,v,w,J]
    plain_loops: List[np.ndarray]  # each is (5, T)
    iteration_times: List[float]
    step_times: List[float]
    casadi_times: List[float]
    dt: float
    augment_extra_points: List[int]
    augment_extra_v_mean: List[float]
    augment_extra_J_mean: List[float]

    @property
    def best_lap_seconds(self) -> float:
        return float(min(loop.shape[1] for loop in self.plain_loops) * self.dt)

    @property
    def best_loop_index(self) -> int:
        lengths = [loop.shape[1] for loop in self.plain_loops]
        return int(np.argmin(lengths))


def _with_time_row(X_log: np.ndarray) -> np.ndarray:
    """
    Convert (5,T) state log to (6,T) by appending a "steps-to-go" row.
    """
    if X_log.ndim != 2 or X_log.shape[0] != 5:
        raise ValueError("X_log must have shape (5, T).")
    T = X_log.shape[1]
    return np.vstack((X_log, np.arange(T)[::-1]))


def _select_start_indices(T: int, n: int) -> np.ndarray:
    if T <= 1:
        return np.array([0], dtype=int)
    n = max(1, min(int(n), T))
    # spread across the lap (skip the last few points)
    idxs = np.linspace(0, max(0, T - 2), n, dtype=int)
    return np.unique(idxs)


def run_lmpc_iterations(
    *,
    M_lmpc: Callable,
    F_model: Callable,
    traj_xytheta: np.ndarray,
    inside_xy: np.ndarray,
    outside_xy: np.ndarray,
    x_init: np.ndarray,
    idx_init: int,
    X_log_first_loop: np.ndarray,
    config: LMPCRunConfig,
    n_iterations: int,
    augmenter: Optional[
        Callable[
            [np.ndarray, np.ndarray, np.ndarray, np.ndarray, LMPCAugmentConfig],
            np.ndarray,
        ]
    ] = None,
    augment_cfg: LMPCAugmentConfig = LMPCAugmentConfig(enabled=False),
) -> LMPCRunResult:
    """
    Runs LMPC iterations and returns logs.

    Parameters expected to match the notebook:
    - `traj_xytheta` is `(T,3)` with columns `[x,y,theta_ref]`.
    - `inside_xy`, `outside_xy` are `(T,2)` boundary points.
    - `M_lmpc(x0, D, J, t_to_N, margins)` returns `(u0, lambda)` like the notebook.

    `augmenter` (optional) is called once per iteration *after* producing the lap,
    and should return an augmented `(6,T_aug)` array compatible with `loops_with_time`.
    """
    from sklearn.neighbors import NearestNeighbors
    from scipy import spatial
    from casadi import DM

    dt = float(config.dt)
    traj_xytheta = np.asarray(traj_xytheta, dtype=float)
    traj_kdtree = spatial.KDTree(traj_xytheta[:, :2])
    start_traj_index = int(config.start_traj_index) if config.start_traj_index is not None else None
    traj_len = int(config.traj_len) if config.traj_len is not None else traj_xytheta.shape[0]
    wrap_thresh = int(max(1, traj_len // 2))

    X_log_orig = np.asarray(X_log_first_loop, dtype=float)
    # Replicate first loop i_j times to initialize safe set, as in notebook:
    # `last_iterations = np.hstack([all_points]*i_j)`
    loops_with_time: List[np.ndarray] = [_with_time_row(X_log_orig)] * config.i_j
    plain_loops: List[np.ndarray] = [X_log_orig]

    # state and reference pointer
    x = np.asarray(x_init, dtype=float).reshape(-1)
    idx = int(idx_init)

    kdins = spatial.KDTree(np.asarray(inside_xy, dtype=float))
    ins_len = inside_xy.shape[0]

    step_times: List[float] = [0.0]
    casadi_times: List[float] = [0.0]
    iteration_times: List[float] = [0.0]
    augment_extra_points: List[int] = []
    augment_extra_v_mean: List[float] = []
    augment_extra_J_mean: List[float] = []

    for iteration in range(int(n_iterations)):
        it_start = timer()

        last_loop = plain_loops[-1]
        kdtree_last_loop = spatial.KDTree(last_loop[:2, : config.frame_rate * 10].T)

        X_log = np.empty((5, 0), dtype=float)
        U_log = np.empty((2, 0), dtype=float)

        # Normalize angle to [-π, π] as in the notebook
        x[2] = ((x[2] + np.pi) % (2 * np.pi)) - np.pi

        # replicate notebook behavior: last i_j loops as safe set
        if config.i_j_all:
            last_iterations = np.hstack(loops_with_time)
        else:
            last_iterations = np.hstack(loops_with_time[-config.i_j :])

        # method 1 from notebook: hide points with small J
        last_iterations_filtered = last_iterations[np.vstack([last_iterations[-1] > config.K] * 6)].reshape(6, -1)

        nbrs = NearestNeighbors(n_neighbors=config.K * config.i_j, algorithm="ball_tree").fit(
            last_iterations_filtered[:2].T
        )

        _, idx = kdtree_last_loop.query(np.array([x[0], x[1]]).reshape(-1), workers=-1)
        t = 0
        already_changed = False

        # Lap completion tracking along centerline index (wrap-aware).
        laps = 0
        last_traj_idx = start_traj_index
        if last_traj_idx is None:
            _, last_traj_idx = traj_kdtree.query(np.array([x[0], x[1]]).reshape(-1), workers=-1)
            last_traj_idx = int(last_traj_idx)
        start_unwrapped = int(last_traj_idx)
        last_unwrapped = int(last_traj_idx)

        while True:
            start_step = timer()

            distances, indices = nbrs.kneighbors([last_loop[:2, (idx + config.N) % last_loop.shape[1]].T])
            indices = indices.reshape(-1)

            if t == 0:
                D = last_iterations_filtered[:-1, indices]
                J = last_iterations_filtered[-1, indices].reshape(-1)
            else:
                # use previous lambda `l` to compute S and resample neighbors
                S = last_iterations_filtered[:, indices] @ l
                distances, indices = nbrs.kneighbors(np.array(S[:2].T))
                indices = indices.reshape(-1)
                D = last_iterations_filtered[:-1, indices]
                J = last_iterations_filtered[-1, indices].reshape(-1)

            _, border_idx = kdins.query(np.array([x[0], x[1]]).reshape(-1), workers=-1)
            margins = (
                np.array(
                    [
                        inside_xy[border_idx],
                        inside_xy[(border_idx + config.N + config.more) % ins_len],
                        outside_xy[border_idx],
                        outside_xy[(border_idx + config.N + int(config.more / 4)) % ins_len],
                        outside_xy[(border_idx + config.N + config.more) % ins_len],
                    ]
                )
                .T.astype(float)
            )

            casadi_timer_start = timer()
            u0, l = M_lmpc(
                x.reshape(5, 1),
                DM(D[:, :]),
                DM(J) / 600,
                (np.arange(t, t + config.N - 1) >= config.finish_t_steps).T,
                margins,
            )
            casadi_timer_end = timer()

            # log
            U_log = np.column_stack((U_log, np.asarray(u0).reshape(2)))
            X_log = np.column_stack((X_log, x.reshape(5)))

            # simulate
            x = np.asarray(F_model(x, np.asarray(u0).reshape(2))).reshape(-1)

            # Update lap progress along centerline index.
            _, traj_idx = traj_kdtree.query(np.array([x[0], x[1]]).reshape(-1), workers=-1)
            traj_idx = int(traj_idx)
            if traj_idx < int(last_traj_idx) - wrap_thresh:
                laps += 1
            elif traj_idx > int(last_traj_idx) + wrap_thresh:
                # Likely wrapped backwards; treat as going backwards across the seam.
                laps -= 1
            last_traj_idx = traj_idx
            last_unwrapped = traj_idx + laps * traj_len

            _, idx_new = kdtree_last_loop.query(np.array([x[0], x[1]]).reshape(-1), workers=-1)
            if idx_new >= idx:
                idx = idx_new
                if idx + config.N > traj_xytheta.shape[0]:
                    idx = traj_xytheta.shape[0] - idx

            # Stop after a full lap (preferred), or fall back to legacy finish-line heuristics.
            if (t >= config.finish_t_steps) and ((last_unwrapped - start_unwrapped) >= traj_len):
                break
            if (start_traj_index is None) and (config.traj_len is None):
                if (x[1] >= config.finish_y) and (t >= config.finish_t_steps) and (x[0] >= config.finish_x_min):
                    break

            t += 1

            if (t / config.frame_rate) > config.max_seconds:
                break

            if t == config.K + config.N:
                kdtree_last_loop = spatial.KDTree(last_loop[:2, :].T)
                last_iterations_filtered = last_iterations
                nbrs = NearestNeighbors(n_neighbors=config.K * config.i_j, algorithm="ball_tree").fit(
                    last_iterations[:2, :].T
                )

            if (x[1] > 1) and (x[1] < 1.7) and (x[0] > 1.5) and (not already_changed):
                last_iterations_filtered[-1, : config.N + config.K * config.i_j] = 0
                last_iterations_filtered[:-1, : config.N + 1] = X_log_orig[:, : config.N + 1]
                last_iterations_filtered[2, : config.N + config.K * config.i_j] += 2 * np.pi
                already_changed = True

            end_step = timer()
            step_times.append(end_step - start_step)
            casadi_times.append(casadi_timer_end - casadi_timer_start)

        plain_loops.append(X_log)
        last_points = _with_time_row(X_log)
        base_last_points = last_points

        if augment_cfg.enabled and augmenter is not None:
            last_points = augmenter(last_points, traj_xytheta, inside_xy, outside_xy, augment_cfg)
            extra_n = int(max(0, last_points.shape[1] - base_last_points.shape[1]))
            augment_extra_points.append(extra_n)
            if extra_n > 0:
                extra = last_points[:, base_last_points.shape[1] :]
                v_mean = float(np.mean(extra[3])) if extra.shape[0] > 3 else float("nan")
                J_mean = float(np.mean(extra[-1]))
            else:
                v_mean = float("nan")
                J_mean = float("nan")
            augment_extra_v_mean.append(v_mean)
            augment_extra_J_mean.append(J_mean)

        loops_with_time.append(last_points)

        it_end = timer()
        iteration_times.append(it_end - it_start)

    return LMPCRunResult(
        loops_with_time=loops_with_time,
        plain_loops=plain_loops,
        iteration_times=iteration_times,
        step_times=step_times,
        casadi_times=casadi_times,
        dt=dt,
        augment_extra_points=augment_extra_points,
        augment_extra_v_mean=augment_extra_v_mean,
        augment_extra_J_mean=augment_extra_J_mean,
    )
