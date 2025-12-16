from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .oracle import phase_oracle_diagonal
from .sampler import QuantumSamplerConfig, _build_grover_circuit


@dataclass(frozen=True)
class QuantumNeighborSelectorConfig:
    """Configuration for quantum-based neighbor selection in LMPC."""
    backend: str = "statevector"
    n_iterations: int = 1
    seed: Optional[int] = None
    # J threshold percentile: mark points with J <= percentile of reachable points
    j_percentile: float = 25.0


class QuantumNeighborSelector:
    """
    Use Grover's algorithm to find safe set points that:
    1. Are reachable (within distance threshold from target)
    2. Have minimum J (cost-to-go)

    This replaces the classical K-nearest-neighbors approach with
    quantum amplitude amplification to bias toward low-J points.
    """

    def __init__(self, config: QuantumNeighborSelectorConfig = QuantumNeighborSelectorConfig()):
        self.config = config

    def select_neighbors(
        self,
        safe_set_xy: np.ndarray,
        safe_set_J: np.ndarray,
        target_xy: np.ndarray,
        reachability_radius: float,
        n_neighbors: int,
    ) -> np.ndarray:
        """
        Select n_neighbors points from the safe set using Grover search.

        Args:
            safe_set_xy: (N, 2) array of safe set x,y positions
            safe_set_J: (N,) array of cost-to-go values
            target_xy: (2,) target position to find neighbors around
            reachability_radius: max distance from target to consider reachable
            n_neighbors: number of neighbors to return (K)

        Returns:
            indices: (n_neighbors,) array of safe set indices
        """
        safe_set_xy = np.asarray(safe_set_xy, dtype=float)
        safe_set_J = np.asarray(safe_set_J, dtype=float).reshape(-1)
        target_xy = np.asarray(target_xy, dtype=float).reshape(-1)

        N = safe_set_xy.shape[0]

        # Compute distances from target
        distances = np.linalg.norm(safe_set_xy - target_xy, axis=1)

        # Build cost table: J for reachable points, inf for unreachable
        cost_table = np.full(N, float("inf"), dtype=float)
        reachable_mask = distances <= reachability_radius
        cost_table[reachable_mask] = safe_set_J[reachable_mask]

        # Check if we have enough reachable points
        n_reachable = np.sum(reachable_mask)
        if n_reachable == 0:
            # Fallback: return K nearest by distance (classical behavior)
            nearest_indices = np.argsort(distances)[:n_neighbors]
            return nearest_indices

        if n_reachable <= n_neighbors:
            # Not enough reachable points, return all reachable + nearest unreachable
            reachable_indices = np.where(reachable_mask)[0]
            unreachable_indices = np.where(~reachable_mask)[0]
            unreachable_sorted = unreachable_indices[np.argsort(distances[unreachable_indices])]
            n_needed = n_neighbors - len(reachable_indices)
            return np.concatenate([reachable_indices, unreachable_sorted[:n_needed]])

        # Pad to power of 2 for Grover
        n_qubits = int(np.ceil(np.log2(max(N, 2))))
        padded_size = 2 ** n_qubits

        cost_table_padded = np.full(padded_size, float("inf"), dtype=float)
        cost_table_padded[:N] = cost_table

        # Compute J threshold: percentile of reachable J values
        reachable_J = safe_set_J[reachable_mask]
        j_threshold = float(np.percentile(reachable_J, self.config.j_percentile))

        # Build oracle and run Grover
        oracle_diag = phase_oracle_diagonal(cost_table_padded, j_threshold)

        if self.config.backend == "statevector":
            indices = self._sample_statevector(
                n_qubits=n_qubits,
                oracle_diag=oracle_diag,
                n_samples=n_neighbors * 4,  # oversample then deduplicate
                cost_table=cost_table_padded,
                N_original=N,
            )
        else:
            raise ValueError(f"Unknown backend: {self.config.backend}")

        # Deduplicate and take top K by J
        indices = np.unique(indices)
        # Filter to valid indices and sort by J
        valid_mask = indices < N
        indices = indices[valid_mask]

        if len(indices) == 0:
            # Fallback to classical
            nearest_indices = np.argsort(distances)[:n_neighbors]
            return nearest_indices

        # Sort by J and take best K
        sorted_by_J = indices[np.argsort(safe_set_J[indices])]
        return sorted_by_J[:n_neighbors]

    def _sample_statevector(
        self,
        n_qubits: int,
        oracle_diag: np.ndarray,
        n_samples: int,
        cost_table: np.ndarray,
        N_original: int,
    ) -> np.ndarray:
        """Sample indices using statevector simulation."""
        try:
            from qiskit.quantum_info import Statevector
        except Exception as e:
            raise RuntimeError("statevector backend requires `qiskit` to be installed.") from e

        qc = _build_grover_circuit(
            n_qubits=n_qubits,
            oracle_diag=oracle_diag,
            n_iterations=self.config.n_iterations,
            with_measurements=False,
        )
        sv = Statevector.from_instruction(qc)
        probs = np.asarray(sv.probabilities(), dtype=float)
        probs = probs / probs.sum()

        rng = np.random.default_rng(self.config.seed)
        samples = rng.choice(len(probs), size=n_samples, replace=True, p=probs)

        # Filter to valid original indices
        valid = samples[samples < N_original]
        return valid.astype(int)


def make_quantum_neighbor_selector(
    config: QuantumNeighborSelectorConfig = QuantumNeighborSelectorConfig(),
):
    """Factory function to create a quantum neighbor selector."""
    return QuantumNeighborSelector(config=config)
