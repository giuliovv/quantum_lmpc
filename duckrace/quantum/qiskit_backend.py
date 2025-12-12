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
        from qiskit_ibm_runtime import QiskitRuntimeService, Session
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "IBM Runtime backend requested but qiskit-ibm-runtime is not available."
        ) from e

    # SamplerV2 exists in newer qiskit-ibm-runtime; fall back to Sampler in older versions.
    SamplerCls = None
    try:
        from qiskit_ibm_runtime import SamplerV2 as SamplerCls  # type: ignore
    except Exception:  # pragma: no cover
        try:
            from qiskit_ibm_runtime import Sampler as SamplerCls  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError("Could not import an IBM Runtime Sampler primitive.") from e

    service = QiskitRuntimeService()
    if config.backend_name is not None:
        backend = service.backend(config.backend_name)
    elif config.least_busy:
        backend = service.least_busy(simulator=config.simulator, operational=True)
    else:
        raise ValueError("Provide backend_name or set least_busy=True.")

    rng = np.random.default_rng(seed)
    n_states = 2**n_qubits

    sampled: list[int] = []
    with Session(service=service, backend=backend) as session:
        sampler = SamplerCls(session=session)

        # Try to pass shots in a version-tolerant way.
        try:
            job = sampler.run(circuits, shots=config.shots)
        except TypeError:  # pragma: no cover
            job = sampler.run(circuits)

        result = job.result()

        # Try common result shapes.
        quasi_dists = None
        if hasattr(result, "quasi_dists"):
            quasi_dists = result.quasi_dists
        elif hasattr(result, "quasi_distributions"):
            quasi_dists = result.quasi_distributions
        elif isinstance(result, (list, tuple)) and len(result) > 0 and hasattr(result[0], "quasi_dists"):
            quasi_dists = result[0].quasi_dists

        if quasi_dists is None:  # pragma: no cover
            raise RuntimeError("Sampler result did not expose quasi distributions.")

        # For now we only use the first circuit's distribution.
        probs = _quasi_to_probs(quasi_dists[0], n_states=n_states)
        sampled = rng.choice(n_states, size=n_samples, replace=True, p=probs).tolist()

    return [int(x) for x in sampled]

