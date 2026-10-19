"""
Draw-level binary edge matrix from latent Z (merged MCMC samples).

M_ij = 1 if Z_ij > Z_BINARY_THRESHOLD else 0.

Z_BINARY_THRESHOLD = 0.5  →  stricter than legacy 0.0 (≈ P > σ(0.5) ≈ 0.62 under logistic link).
Set to 0.0 to recover the old rule (equivalent to P > 0.5 when p = σ(Z)).
"""

from __future__ import annotations

import numpy as np

# Single place to change for full post-sampling re-runs.
Z_BINARY_THRESHOLD = 0.5


def effective_z_binary_threshold(override: float | None) -> float:
    if override is None:
        return float(Z_BINARY_THRESHOLD)
    return float(override)


def binary_m_from_z(z: np.ndarray, *, z_threshold: float | None = None) -> np.ndarray:
    """uint8 array, same shape as z."""
    t = effective_z_binary_threshold(z_threshold)
    return (z > t).astype(np.uint8)
