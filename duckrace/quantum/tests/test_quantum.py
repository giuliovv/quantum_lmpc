import numpy as np

from duckrace.quantum.discretization import decode_control_sequence, encode_control_sequence
from duckrace.quantum.oracle import compute_adaptive_threshold, marked_mask, phase_oracle_diagonal
from duckrace.quantum.sampler import QuantumControlSampler, QuantumSamplerConfig


def test_encode_decode_roundtrip():
    horizon = 4
    for idx in range(4**horizon):
        wl, wr = decode_control_sequence(idx, horizon=horizon)
        wl_bits = [int(x) for x in wl]
        wr_bits = [int(x) for x in wr]
        assert encode_control_sequence(wl_bits, wr_bits) == idx


def test_oracle_marks_expected_states():
    costs = np.array([10.0, 5.0, np.inf, 1.0])
    thr = 5.0
    mask = marked_mask(costs, thr)
    assert mask.tolist() == [False, True, False, True]
    diag = phase_oracle_diagonal(costs, thr)
    assert diag.tolist() == [1.0 + 0j, -1.0 + 0j, 1.0 + 0j, -1.0 + 0j]


def test_threshold_is_finite_when_possible():
    costs = np.array([10.0, 5.0, np.inf, 1.0])
    thr = compute_adaptive_threshold(costs, percentile=50)
    assert np.isfinite(thr)


def test_sampler_returns_valid_indices_statevector():
    horizon = 2
    n_qubits = 2 * horizon  # 4 qubits => 16 states
    costs = np.arange(2**n_qubits, dtype=float)
    thr = 3.0
    sampler = QuantumControlSampler(n_qubits=n_qubits, config=QuantumSamplerConfig(backend="statevector", seed=0))
    idxs = sampler.sample(costs, thr, n_samples=20)
    assert len(idxs) == 20
    assert all(isinstance(i, int) and 0 <= i < 2**n_qubits for i in idxs)

