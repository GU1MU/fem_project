from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.sparse import csr_matrix

from .condition import BoundaryCondition


def apply_dirichlet(
    K: csr_matrix,
    F: np.ndarray,
    bc: BoundaryCondition,
) -> Tuple[csr_matrix, np.ndarray]:
    """Apply prescribed displacements by zeroing rows and columns."""
    if not isinstance(K, csr_matrix):
        raise TypeError(f"K must be csr_matrix, got {type(K)}")
    if K.shape[0] != K.shape[1]:
        raise ValueError(f"K must be square, got {K.shape}")

    n = K.shape[0]
    F = np.asarray(F, dtype=float).ravel()
    if F.shape[0] != n:
        raise ValueError(f"F must have length {n}, got {F.shape}")

    K_mod = K.copy().tolil()
    F_mod = F.copy()

    for dof_id, value in bc.prescribed_displacements.items():
        if dof_id < 0 or dof_id >= n:
            raise IndexError(f"DOF index {dof_id} out of bounds [0, {n})")
        value = float(value)
        if value != 0.0:
            F_mod -= value * K_mod[:, dof_id].toarray().ravel()
        K_mod[dof_id, :] = 0.0
        K_mod[:, dof_id] = 0.0
        K_mod[dof_id, dof_id] = 1.0
        F_mod[dof_id] = value

    return K_mod.tocsr(), F_mod
