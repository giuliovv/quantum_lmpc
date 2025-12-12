from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import numpy as np


State = np.ndarray
Control = Tuple[float, float]


def rollout_trajectory(
    x0: State,
    controls: Sequence[Control],
    model_F: Callable,
) -> List[State]:
    """
    Roll out dynamics using a (CasADi) model_F compatible with utils.model_F:
      next_x = model_F(x, u)["dae"]  OR  model_F(x, u) -> next_x
    """
    trajectory: List[State] = [np.asarray(x0, dtype=float).reshape(-1)]

    for (wl, wr) in controls:
        x = trajectory[-1]
        u = np.asarray([wl, wr], dtype=float).reshape(-1)
        nxt = model_F(x, u)
        if isinstance(nxt, dict) and "dae" in nxt:
            nxt = nxt["dae"]
        nxt = np.asarray(nxt, dtype=float).reshape(-1)
        trajectory.append(nxt)
    return trajectory


def compute_cost(
    trajectory: Sequence[State],
    controls: Sequence[Control],
    centerline_xy: np.ndarray,
    alpha: float = 1.0,
    beta: float = 0.1,
    gamma: float = 0.0,
) -> float:
    """
    Minimal, model-agnostic cost:
    - terminal distance to closest point on centerline
    - small control effort regularization
    - optional speed penalty (if state has v at index 3)
    """
    if len(trajectory) == 0:
        return float("inf")

    xT = np.asarray(trajectory[-1], dtype=float).reshape(-1)
    pos = xT[:2]
    cl = np.asarray(centerline_xy, dtype=float).reshape(-1, 2)
    terminal_dist = float(np.min(np.sum((cl - pos) ** 2, axis=1)))

    effort = float(np.sum(np.sum(np.asarray(controls, dtype=float) ** 2, axis=1)))

    speed_pen = 0.0
    if xT.size > 3:
        v = float(xT[3])
        speed_pen = max(0.0, v - 1.0) ** 2

    return alpha * terminal_dist + beta * effort + gamma * speed_pen


def is_feasible(
    trajectory: Sequence[State],
    track_bounds: Optional[Callable[[State], bool]] = None,
) -> bool:
    """
    Feasibility predicate.

    For the minimal prototype we accept a callable `track_bounds(state)->bool`.
    If `track_bounds` is None, everything is feasible.
    """
    if track_bounds is None:
        return True

    for x in trajectory:
        if not bool(track_bounds(np.asarray(x, dtype=float).reshape(-1))):
            return False
    return True


def build_cost_table(
    x0: State,
    model_F: Callable,
    horizon: int,
    centerline_xy: np.ndarray,
    track_bounds: Optional[Callable[[State], bool]] = None,
    cost_fn: Callable[[Sequence[State], Sequence[Control], np.ndarray], float] = compute_cost,
    decode_fn: Optional[Callable[[int, int], Tuple[List[float], List[float]]]] = None,
) -> np.ndarray:
    """
    Precompute costs for all 4^T control sequences.
    Infeasible sequences get cost = +inf.
    """
    from .discretization import decode_control_sequence

    if decode_fn is None:
        decode_fn = decode_control_sequence

    n_total = 4**horizon
    costs = np.full((n_total,), float("inf"), dtype=float)

    for idx in range(n_total):
        wl, wr = decode_fn(idx, horizon)
        controls = list(zip(wl, wr))
        traj = rollout_trajectory(x0=x0, controls=controls, model_F=model_F)
        if not is_feasible(traj, track_bounds=track_bounds):
            continue
        costs[idx] = float(cost_fn(traj, controls, centerline_xy))

    return costs

