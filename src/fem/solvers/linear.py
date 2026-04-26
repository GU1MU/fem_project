from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve


def solve(K: csr_matrix, F: np.ndarray) -> np.ndarray:
    """Solve sparse linear system K @ U = F."""
    if not isinstance(K, csr_matrix):
        raise TypeError(f"K must be csr_matrix, got {type(K)}")
    if K.shape[0] != K.shape[1]:
        raise ValueError(f"K must be square, got {K.shape}")

    n = K.shape[0]
    F = np.asarray(F, dtype=float)
    if F.ndim == 2 and F.shape[1] == 1:
        F = F.ravel()
    if F.ndim != 1 or F.shape[0] != n:
        raise ValueError(f"F must have length {n}, got {F.shape}")

    try:
        return spsolve(K, F)
    except Exception as exc:
        raise RuntimeError(
            f"sparse linear solve failed: {exc}. "
            "The stiffness matrix may be singular or under-constrained."
        ) from exc
