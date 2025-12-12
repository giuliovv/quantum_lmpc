from __future__ import annotations

import argparse
import sys
from dataclasses import replace

import numpy as np

from duckrace.quantum.integration import QuantumSafeSetExpander
from duckrace.quantum.qiskit_backend import IBMRuntimeConfig
from duckrace.quantum.sampler import QuantumSamplerConfig


def _build_centerline(n: int = 50, length: float = 5.0) -> np.ndarray:
    x = np.linspace(0.0, float(length), int(n))
    y = np.zeros_like(x)
    return np.stack([x, y], axis=1)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m duckrace.quantum",
        description="Run the minimal quantum control sampler demo.",
    )
    parser.add_argument("--backend", choices=["statevector", "ibm_runtime"], default="statevector")
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ibm-backend-name", type=str, default=None)
    parser.add_argument("--ibm-least-busy", action="store_true", default=True)
    parser.add_argument("--ibm-simulator", action="store_true", default=False)
    parser.add_argument("--ibm-shots", type=int, default=1024)
    args = parser.parse_args(argv)

    try:
        import utils
    except Exception as e:
        raise RuntimeError(
            "Could not import `utils.py` from repo root. Run this from the repo root "
            "(so `utils.py` is on PYTHONPATH), or adapt the entrypoint to your project."
        ) from e

    model_F = utils.model_F(dt=float(args.dt))

    sampler_cfg = QuantumSamplerConfig(
        backend=args.backend,
        n_iterations=int(args.iterations),
        seed=int(args.seed) if args.seed is not None else None,
        ibm=IBMRuntimeConfig(
            backend_name=args.ibm_backend_name,
            least_busy=bool(args.ibm_least_busy),
            simulator=bool(args.ibm_simulator),
            shots=int(args.ibm_shots),
        ),
    )

    expander = QuantumSafeSetExpander(model_F=model_F, horizon=int(args.horizon), sampler_config=sampler_cfg)

    x0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    centerline = _build_centerline()

    trajectories = expander.expand_safe_set(x0=x0, centerline_xy=centerline, n_samples=int(args.samples))
    last_states = np.asarray([traj[-1] for traj in trajectories], dtype=float)

    print(f"backend={args.backend} horizon={args.horizon} iterations={args.iterations} samples={args.samples}")
    print("last_states (x, y, theta, v, w):")
    np.set_printoptions(precision=4, suppress=True)
    print(last_states)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

