from __future__ import annotations

import numpy as np


def plane_stress_matrix(E: float, nu: float) -> np.ndarray:
    """Return isotropic plane stress constitutive matrix."""
    E = float(E)
    nu = float(nu)
    coef = E / (1.0 - nu ** 2)
    return coef * np.array([
        [1.0,    nu,           0.0],
        [nu,     1.0,          0.0],
        [0.0,    0.0, (1.0 - nu) / 2.0],
    ], dtype=float)


def plane_strain_matrix(E: float, nu: float) -> np.ndarray:
    """Return isotropic plane strain constitutive matrix."""
    E = float(E)
    nu = float(nu)
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))

    return np.array([
        [lam + 2.0 * mu, lam,            0.0],
        [lam,            lam + 2.0 * mu, 0.0],
        [0.0,            0.0,            mu],
    ], dtype=float)


def plane_matrix(E: float, nu: float, plane_type: str) -> np.ndarray:
    """Return plane stress or plane strain constitutive matrix."""
    pt = str(plane_type).lower()
    if pt.startswith("stress"):
        return plane_stress_matrix(E, nu)
    if pt.startswith("strain"):
        return plane_strain_matrix(E, nu)
    raise ValueError(f"invalid plane_type={plane_type}")


def solid_3d_matrix(E: float, nu: float) -> np.ndarray:
    """Return isotropic 3D constitutive matrix."""
    E = float(E)
    nu = float(nu)
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))

    return np.array([
        [lam + 2.0 * mu, lam,            lam,            0.0, 0.0, 0.0],
        [lam,            lam + 2.0 * mu, lam,            0.0, 0.0, 0.0],
        [lam,            lam,            lam + 2.0 * mu, 0.0, 0.0, 0.0],
        [0.0,            0.0,            0.0,            mu,  0.0, 0.0],
        [0.0,            0.0,            0.0,            0.0, mu,  0.0],
        [0.0,            0.0,            0.0,            0.0, 0.0, mu],
    ], dtype=float)
