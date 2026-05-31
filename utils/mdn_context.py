"""
MDN context vector builder for SubRep.

Formalizes the 14D context layout consumed by MotiveDecompositionNetwork:

    context = [obs(8D), delta_r(1D), delta_n(2D), gate_indicator(2D), margin(1D)]

This ensures all callers construct contexts identically.
"""

from __future__ import annotations

import numpy as np


CONTEXT_DIM = 14

_GATE_VECTORS = {
    "CDS": np.array([1.0, 0.0], dtype=np.float32),
    "PDS": np.array([0.0, 1.0], dtype=np.float32),
}


def build_mdn_context(
    obs: np.ndarray,
    delta_r: float,
    delta_n: tuple[float, float] | np.ndarray,
    gate_type: str,
    admission_margin: float,
) -> np.ndarray:
    """
    Build a 14D MDN context vector from skill evaluation data.

    Layout:
        [0:8]   obs             — environment observation
        [8]     delta_r         — scalar payoff improvement
        [9:11]  delta_n         — motive improvement vector (2D)
        [11:13] gate_indicator  — one-hot: [1,0]=CDS, [0,1]=PDS
        [13]    margin          — admission margin

    Args:
        obs: Environment observation, shape (8,).
        delta_r: Scalar payoff improvement over baseline.
        delta_n: 2D motive improvement vector.
        gate_type: "CDS" or "PDS".
        admission_margin: How far above the gate threshold.

    Returns:
        1D float32 array of shape (14,).

    Raises:
        ValueError: If inputs have wrong shape or gate_type is invalid.
    """
    obs = np.asarray(obs, dtype=np.float32).reshape(-1)
    if obs.shape[0] != 8:
        raise ValueError(f"obs must have 8 elements, got {obs.shape[0]}")

    delta_n = np.asarray(delta_n, dtype=np.float32).reshape(-1)
    if delta_n.shape[0] != 2:
        raise ValueError(f"delta_n must have 2 elements, got {delta_n.shape[0]}")

    gate_type_upper = gate_type.strip().upper()
    if gate_type_upper not in _GATE_VECTORS:
        raise ValueError(f"gate_type must be 'CDS' or 'PDS', got {gate_type!r}")

    gate_vec = _GATE_VECTORS[gate_type_upper]

    context = np.concatenate([
        obs,                                        # 8 elements
        np.array([float(delta_r)], dtype=np.float32),  # 1 element
        delta_n,                                    # 2 elements
        gate_vec,                                   # 2 elements
        np.array([float(admission_margin)], dtype=np.float32),  # 1 element
    ])

    assert context.shape == (CONTEXT_DIM,), f"Context shape mismatch: {context.shape}"
    return context


def build_mdn_context_from_entry(obs: np.ndarray, entry) -> np.ndarray:
    """
    Build a 14D context from an observation and a SkillEntry.

    Convenience wrapper that extracts delta_r, delta_n, gate_type,
    and admission_margin from the SkillEntry's certificate.

    Args:
        obs: Environment observation, shape (8,).
        entry: A SkillEntry from the SkillLibrary.

    Returns:
        1D float32 array of shape (14,).
    """
    return build_mdn_context(
        obs=obs,
        delta_r=entry.delta_r,
        delta_n=entry.delta_n,
        gate_type=entry.gate_type,
        admission_margin=entry.admission_margin,
    )
