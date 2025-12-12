from __future__ import annotations

def diffusion_operator(n_qubits: int):
    """
    Standard Grover diffusion operator: 2|s><s| - I
    Implemented as H^n X^n (MCZ) X^n H^n.
    """
    try:
        from qiskit import QuantumCircuit
    except Exception as e:  # pragma: no cover
        raise RuntimeError("diffusion_operator requires `qiskit` to be installed.") from e

    if n_qubits <= 0:
        raise ValueError("n_qubits must be > 0.")

    qc = QuantumCircuit(n_qubits, name="Diffusion")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))

    if n_qubits == 1:
        qc.z(0)
    else:
        target = n_qubits - 1
        controls = list(range(n_qubits - 1))
        qc.h(target)
        qc.mcx(controls, target)
        qc.h(target)

    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def optimal_grover_iterations(n_total: int, n_marked: int) -> int:
    """
    Approximate optimal Grover iterations for amplitude amplification.
    For the minimal prototype we usually use 1 iteration.
    """
    if n_total <= 0:
        raise ValueError("n_total must be > 0.")
    if n_marked <= 0:
        return 0
    import math

    theta = math.asin(math.sqrt(n_marked / n_total))
    return max(1, int(round((math.pi / (4 * theta)) - 0.5)))
