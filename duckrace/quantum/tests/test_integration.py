import numpy as np

import utils
from duckrace.quantum.integration import QuantumSafeSetExpander
from duckrace.quantum.sampler import QuantumSamplerConfig


def test_quantum_sampling_runs_end_to_end_statevector():
    dt = 0.05
    model_F = utils.model_F(dt=dt)
    expander = QuantumSafeSetExpander(model_F=model_F, horizon=3, sampler_config=QuantumSamplerConfig(backend="statevector", seed=0))

    x0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    centerline = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])

    trajectories = expander.expand_safe_set(x0=x0, centerline_xy=centerline, n_samples=5)
    assert len(trajectories) == 5
    assert all(len(traj) == expander.horizon + 1 for traj in trajectories)

