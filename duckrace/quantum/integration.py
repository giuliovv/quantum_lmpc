from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from .cost_function import build_cost_table, rollout_trajectory
from .discretization import decode_control_sequence
from .oracle import compute_adaptive_threshold
from .sampler import QuantumControlSampler, QuantumSamplerConfig


State = np.ndarray


@dataclass(frozen=True)
class QuantumSafeSetExpander:
    model_F: Callable
    horizon: int = 4
    sampler_config: QuantumSamplerConfig = QuantumSamplerConfig()

    def __post_init__(self):
        if self.horizon <= 0:
            raise ValueError("horizon must be > 0.")

    @property
    def n_qubits(self) -> int:
        return 2 * self.horizon

    def expand_safe_set(
        self,
        x0: State,
        centerline_xy: np.ndarray,
        track_bounds: Optional[Callable[[State], bool]] = None,
        n_samples: int = 10,
        threshold: Optional[float] = None,
    ) -> List[List[State]]:
        cost_table = build_cost_table(
            x0=x0,
            model_F=self.model_F,
            horizon=self.horizon,
            centerline_xy=centerline_xy,
            track_bounds=track_bounds,
        )

        if threshold is None:
            threshold = compute_adaptive_threshold(cost_table, percentile=25.0)

        sampler = QuantumControlSampler(n_qubits=self.n_qubits, config=self.sampler_config)
        indices = sampler.sample(cost_table=cost_table, threshold=threshold, n_samples=n_samples)

        trajectories: List[List[State]] = []
        for idx in indices:
            wl, wr = decode_control_sequence(int(idx), self.horizon)
            controls = list(zip(wl, wr))
            traj = rollout_trajectory(x0=x0, controls=controls, model_F=self.model_F)
            trajectories.append(traj)
        return trajectories

    def sample_random(
        self,
        x0: State,
        centerline_xy: np.ndarray,
        track_bounds: Optional[Callable[[State], bool]] = None,
        n_samples: int = 10,
        seed: Optional[int] = None,
    ) -> List[List[State]]:
        rng = np.random.default_rng(seed)
        n_total = 4**self.horizon
        indices = rng.integers(0, n_total, size=n_samples).tolist()
        trajectories: List[List[State]] = []
        for idx in indices:
            wl, wr = decode_control_sequence(int(idx), self.horizon)
            controls = list(zip(wl, wr))
            traj = rollout_trajectory(x0=x0, controls=controls, model_F=self.model_F)
            trajectories.append(traj)
        return trajectories

