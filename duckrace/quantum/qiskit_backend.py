from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class IBMRuntimeConfig:
    backend_name: Optional[str] = None
    least_busy: bool = True
    simulator: bool = False
    shots: int = 1024


def _quasi_to_probs(quasi: Any, n_states: int) -> np.ndarray:
    probs = np.zeros((n_states,), dtype=float)
    if isinstance(quasi, dict):
        items = quasi.items()
    else:
        items = getattr(quasi, "items", lambda: [])()
    for k, v in items:
        probs[int(k)] = float(v)
    probs = np.clip(probs, 0.0, None)
    s = probs.sum()
    if s <= 0:
        probs[:] = 1.0 / n_states
    else:
        probs /= s
    return probs


def sample_indices_ibm_runtime(
    circuits: Sequence,
    n_qubits: int,
    n_samples: int,
    config: IBMRuntimeConfig,
    seed: Optional[int] = None,
) -> list[int]:
    """
    Run circuits on IBM Runtime using the Sampler primitive and return sampled basis indices.

    Requires `qiskit-ibm-runtime` and a configured account (see `connection_test.ipynb`).
    """
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "IBM Runtime backend requested but qiskit-ibm-runtime is not available."
        ) from e

    service = QiskitRuntimeService()
    if config.backend_name is not None:
        backend = service.backend(config.backend_name)
    elif config.least_busy:
        backend = service.least_busy(simulator=config.simulator, operational=True)
    else:
        raise ValueError("Provide backend_name or set least_busy=True.")

    rng = np.random.default_rng(seed)
    n_states = 2**n_qubits

    # Transpile circuits for the target backend
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
    isa_circuits = pm.run(circuits)

    # Run with SamplerV2 (qiskit-ibm-runtime >= 0.21)
    sampler = SamplerV2(mode=backend)
    job = sampler.run(isa_circuits, shots=config.shots)
    result = job.result()

    # Extract counts from PubResult - newer API returns BitArray in data
    pub_result = result[0]
    if hasattr(pub_result, "data"):
        # Find the measurement register (usually 'meas' or 'c')
        data = pub_result.data
        bit_array = None
        for attr in ["meas", "c"] + [k for k in dir(data) if not k.startswith("_")]:
            candidate = getattr(data, attr, None)
            if candidate is not None and hasattr(candidate, "get_counts"):
                bit_array = candidate
                break
        if bit_array is None:
            raise RuntimeError(f"Could not find measurement data in result: {dir(data)}")
        counts = bit_array.get_counts()
    elif hasattr(pub_result, "quasi_dists"):
        # Fallback for older result format
        probs = _quasi_to_probs(pub_result.quasi_dists[0], n_states=n_states)
        return rng.choice(n_states, size=n_samples, replace=True, p=probs).tolist()
    else:
        raise RuntimeError(f"Unexpected result format: {type(pub_result)}")

    # Convert counts dict (bitstring -> count) to probability distribution
    probs = np.zeros((n_states,), dtype=float)
    total = 0
    for bitstring, count in counts.items():
        # Bitstring is in little-endian format from qiskit, convert to int
        idx = int(bitstring, 2)
        probs[idx] = float(count)
        total += count
    if total > 0:
        probs /= total
    else:
        probs[:] = 1.0 / n_states

    sampled = rng.choice(n_states, size=n_samples, replace=True, p=probs).tolist()
    return [int(x) for x in sampled]

