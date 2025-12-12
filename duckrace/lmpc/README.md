# LMPC Runner + Quantum Comparison

This folder provides a small Python runner that mirrors the LMPC loop used in `iterative_mpc_hull_constraint.ipynb`, plus an optional hook to augment the safe set between iterations.

## Intended usage (from the notebook)

After you have the usual notebook variables computed (`M`, `F`, `traj`, `inside`, `outside`, `finish_line`, `X_log_orig`, `x`, `idx`, `N`, `K`, `i_j`, `more`, `dt`, `env.frame_rate`), you can compare baseline LMPC vs quantum-augmented LMPC:

```python
from casadi import Function

from duckrace.lmpc import run_lmpc_iterations, LMPCRunConfig, LMPCAugmentConfig
from duckrace.quantum.lmpc_augment import (
    QuantumLMPCAugmenterConfig,
    make_lmpc_runner_augmenter,
)

M_lmpc = Function.load("LMPC.casadi")

run_cfg = LMPCRunConfig(
    N=N,
    K=K,
    i_j=i_j,
    frame_rate=env.frame_rate,
    more=more,
    dt=dt,
    finish_y=finish_line.y,
    finish_x_min=finish_line.x - 0.1,
    finish_t_steps=finish_line.t,
)

# Baseline
baseline = run_lmpc_iterations(
    M_lmpc=M_lmpc,
    F_model=F,
    traj_xytheta=traj,
    inside_xy=inside,
    outside_xy=outside,
    x_init=np.array(x).reshape(-1),
    idx_init=int(idx),
    X_log_first_loop=X_log_orig,
    config=run_cfg,
    n_iterations=5,
)

# Quantum-augmented safe-set
q_aug = make_lmpc_runner_augmenter(
    model_F=F,
    cfg=QuantumLMPCAugmenterConfig(
        horizon=4,
        n_start_states=8,
        n_samples_per_start=4,
    ),
)

quantum = run_lmpc_iterations(
    M_lmpc=M_lmpc,
    F_model=F,
    traj_xytheta=traj,
    inside_xy=inside,
    outside_xy=outside,
    x_init=np.array(x).reshape(-1),
    idx_init=int(idx),
    X_log_first_loop=X_log_orig,
    config=run_cfg,
    n_iterations=5,
    augmenter=q_aug,
    augment_cfg=LMPCAugmentConfig(enabled=True),
)

print("baseline best lap (s):", baseline.best_lap_seconds)
print("quantum   best lap (s):", quantum.best_lap_seconds)
```

Notes:
- The quantum augmenter uses Qiskit statevector sampling by default; for IBM Q, configure `QuantumSamplerConfig(backend="ibm_runtime", ...)` inside `QuantumLMPCAugmenterConfig`.
- The feasibility check is approximate (distance to centerline <= half track width at nearest index). If you see unsafe points being added, tighten `feasibility_tol`.

## One-Command Script

From the repo root you can run a full baseline comparison (and optionally quantum):

```bash
python3 -m duckrace.lmpc.duckietown_compare --iterations 5
python3 -m duckrace.lmpc.duckietown_compare --iterations 5 --quantum --quantum-backend statevector
python3 -m duckrace.lmpc.duckietown_compare --iterations 5 --quantum --plot --plot-out assets/lmpc_compare.png
python3 -m duckrace.lmpc.duckietown_compare --iterations 5 --quantum --diagnostics
```
