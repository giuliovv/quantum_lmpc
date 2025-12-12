from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np

from .grover import diffusion_operator
from .oracle import phase_oracle_diagonal
from .qiskit_backend import IBMRuntimeConfig, sample_indices_ibm_runtime


BackendKind = Literal["statevector", "ibm_runtime"]


@dataclass(frozen=True)
class QuantumSamplerConfig:
    backend: BackendKind = "statevector"
    n_iterations: int = 1
    seed: Optional[int] = None
    ibm: IBMRuntimeConfig = IBMRuntimeConfig()


def _build_grover_circuit(
    n_qubits: int,
    oracle_diag: np.ndarray,
    n_iterations: int,
    with_measurements: bool,
):
    try:
        from qiskit import QuantumCircuit
        from qiskit.circuit.library import Diagonal
    except Exception as e:  # pragma: no cover
        raise RuntimeError("_build_grover_circuit requires `qiskit` to be installed.") from e

    if n_qubits <= 0:
        raise ValueError("n_qubits must be > 0.")
    if oracle_diag.shape != (2**n_qubits,):
        raise ValueError("oracle diagonal has wrong shape for n_qubits.")
    if n_iterations < 0:
        raise ValueError("n_iterations must be >= 0.")

    qc = QuantumCircuit(n_qubits, n_qubits if with_measurements else 0)
    qc.h(range(n_qubits))

    oracle_gate = Diagonal(oracle_diag.tolist()).to_gate(label="Oracle")
    diffusion_gate = diffusion_operator(n_qubits).to_gate()

    for _ in range(n_iterations):
        qc.append(oracle_gate, list(range(n_qubits)))
        qc.append(diffusion_gate, list(range(n_qubits)))

    if with_measurements:
        qc.measure(range(n_qubits), range(n_qubits))
    return qc


class QuantumControlSampler:
    """
    Quantum amplitude amplification sampler over control sequence indices.

    Minimal prototype:
    - binary controls, horizon T=4 => 8 qubits (2 per step)
    - precomputed cost table of size 2^n
    - single Grover iteration by default

    Backends:
    - `statevector`: pure simulation (no Qiskit Aer needed)
    - `ibm_runtime`: IBM Q via `qiskit-ibm-runtime` (see connection_test.ipynb)
    """

    def __init__(self, n_qubits: int, config: QuantumSamplerConfig = QuantumSamplerConfig()):
        self.n_qubits = int(n_qubits)
        self.config = config

    def sample(self, cost_table: np.ndarray, threshold: float, n_samples: int = 10) -> list[int]:
        oracle_diag = phase_oracle_diagonal(cost_table=cost_table, threshold=threshold)

        if self.config.backend == "statevector":
            try:
                from qiskit.quantum_info import Statevector
            except Exception as e:  # pragma: no cover
                raise RuntimeError("statevector backend requires `qiskit` to be installed.") from e

            qc = _build_grover_circuit(
                n_qubits=self.n_qubits,
                oracle_diag=oracle_diag,
                n_iterations=self.config.n_iterations,
                with_measurements=False,
            )
            sv = Statevector.from_instruction(qc)
            probs = np.asarray(sv.probabilities(), dtype=float)
            probs = probs / probs.sum()
            rng = np.random.default_rng(self.config.seed)
            return rng.choice(2**self.n_qubits, size=n_samples, replace=True, p=probs).astype(int).tolist()

        if self.config.backend == "ibm_runtime":
            qc = _build_grover_circuit(
                n_qubits=self.n_qubits,
                oracle_diag=oracle_diag,
                n_iterations=self.config.n_iterations,
                with_measurements=True,
            )
            return sample_indices_ibm_runtime(
                circuits=[qc],
                n_qubits=self.n_qubits,
                n_samples=n_samples,
                config=self.config.ibm,
                seed=self.config.seed,
            )

        raise ValueError(f"Unknown backend: {self.config.backend}")
