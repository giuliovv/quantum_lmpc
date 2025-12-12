# Quantum-Enhanced Safe-Set Expansion (Minimal Prototype)

This package implements a minimal quantum amplitude amplification sampler over discretized control sequences. It can be imported from notebooks, or run directly as a module for a quick smoke-test.

## Run It

From the repo root:

```bash
python3 -m duckrace.quantum --backend statevector --horizon 4 --iterations 1 --samples 10
```

## Simulator-first

The default backend is pure simulation via Qiskit Statevector (no Aer needed):

```python
import numpy as np
import utils
from duckrace.quantum import QuantumSafeSetExpander, QuantumSamplerConfig

F = utils.model_F(dt=0.05)
expander = QuantumSafeSetExpander(
    model_F=F,
    horizon=4,
    sampler_config=QuantumSamplerConfig(backend="statevector", n_iterations=1, seed=0),
)

x0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
centerline = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
trajectories = expander.expand_safe_set(x0=x0, centerline_xy=centerline, n_samples=10)
```

## IBM Q (ready to plug in)

To run the same circuit via IBM Runtime, switch the backend:

```python
from duckrace.quantum import QuantumSafeSetExpander, QuantumSamplerConfig
from duckrace.quantum.qiskit_backend import IBMRuntimeConfig

cfg = QuantumSamplerConfig(
    backend="ibm_runtime",
    n_iterations=1,
    ibm=IBMRuntimeConfig(backend_name=None, least_busy=True, simulator=False, shots=1024),
)
expander = QuantumSafeSetExpander(model_F=F, horizon=4, sampler_config=cfg)
```

This uses `qiskit-ibm-runtime`. For credentials, set your token in the environment (for example via a local `.env` file) and run `QiskitRuntimeService.save_account(token=os.environ["QISKIT_IBM_TOKEN"], ...)` as shown in `connection_test.ipynb`.

## LMPC (Duckietown) Comparison

If you’re working in `iterative_mpc_hull_constraint.ipynb`, you can compare baseline LMPC vs quantum-augmented LMPC safe-set expansion using the helper in `duckrace/lmpc/README.md:1`.

## Dependencies

- `qiskit` (required)
- `qiskit-ibm-runtime` (only for IBM Q runs)
- `pytest` (only for tests)
