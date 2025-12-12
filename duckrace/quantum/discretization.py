from __future__ import annotations

from typing import List, Tuple

CONTROL_VALUES: List[float] = [0.0, 1.0]
QUBITS_PER_STEP: int = 2  # (wl, wr) each 1 bit
ACTIONS_PER_STEP: int = 4  # 2 x 2 combinations


def get_control_values() -> List[float]:
    return list(CONTROL_VALUES)


def encode_control_sequence(wl_values: List[int], wr_values: List[int]) -> int:
    """
    Encode a (wl, wr) sequence into an integer index in [0, 4^T).

    Conventions:
    - wl, wr are binary (0/1).
    - action_id(t) = wl(t) + 2 * wr(t)  in [0..3]
    - index = sum_t action_id(t) * 4^t  (little-endian over time)
    """
    if len(wl_values) != len(wr_values):
        raise ValueError("wl_values and wr_values must have the same length.")

    index = 0
    for t, (wl, wr) in enumerate(zip(wl_values, wr_values)):
        if wl not in (0, 1) or wr not in (0, 1):
            raise ValueError("wl/wr must be binary (0/1) for the minimal prototype.")
        action_id = wl + 2 * wr
        index += action_id * (ACTIONS_PER_STEP**t)
    return index


def decode_control_sequence(index: int, horizon: int) -> Tuple[List[float], List[float]]:
    """
    Decode an integer index into (wl, wr) control sequences (floats).
    """
    if horizon <= 0:
        raise ValueError("horizon must be > 0.")
    if index < 0 or index >= (ACTIONS_PER_STEP**horizon):
        raise ValueError(f"index out of range for horizon={horizon}.")

    wl: List[float] = []
    wr: List[float] = []
    remainder = index
    for _ in range(horizon):
        action_id = remainder % ACTIONS_PER_STEP
        remainder //= ACTIONS_PER_STEP
        wl_bit = action_id & 0b01
        wr_bit = (action_id >> 1) & 0b01
        wl.append(CONTROL_VALUES[wl_bit])
        wr.append(CONTROL_VALUES[wr_bit])
    return wl, wr

