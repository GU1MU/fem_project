from __future__ import annotations

import numpy as np


def von_mises_plane(
    sig_x: float,
    sig_y: float,
    tau_xy: float,
    plane_type: str = "stress",
    nu: float = 0.0,
) -> float:
    """Return plane stress or plane strain von Mises stress."""
    sig_x = float(sig_x)
    sig_y = float(sig_y)
    tau_xy = float(tau_xy)
    if str(plane_type).lower().startswith("strain"):
        sig_z = float(nu) * (sig_x + sig_y)
        return float(np.sqrt(
            0.5 * ((sig_x - sig_y)**2 + (sig_y - sig_z)**2 + (sig_z - sig_x)**2)
            + 3.0 * tau_xy**2
        ))
    return float(np.sqrt(sig_x**2 - sig_x * sig_y + sig_y**2 + 3.0 * tau_xy**2))


def von_mises_3d(
    sig_x: float,
    sig_y: float,
    sig_z: float,
    tau_xy: float,
    tau_yz: float,
    tau_zx: float,
) -> float:
    """Return 3D von Mises stress."""
    return float(np.sqrt(
        0.5 * ((sig_x - sig_y)**2 + (sig_y - sig_z)**2 + (sig_z - sig_x)**2)
        + 3.0 * (tau_xy**2 + tau_yz**2 + tau_zx**2)
    ))
