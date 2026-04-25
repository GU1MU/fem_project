import csv
from typing import Sequence


def convert_nodal_solution_into_polar_coord(
    csv_path: str,
    center: Sequence[float],
    out_path: str,
) -> None:
    """Convert nodal displacement or stress CSV into polar components."""
    if len(center) != 2:
        raise ValueError("center must have 2 values")
    cx, cy = float(center[0]), float(center[1])

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("CSV has no header")

        fields = list(reader.fieldnames)
        has_disp = "ux" in fields and "uy" in fields
        has_stress = "sig_x" in fields and "sig_y" in fields and "tau_xy" in fields

        if has_disp and has_stress:
            raise ValueError("CSV has both displacement and stress columns")
        if not has_disp and not has_stress:
            raise ValueError("CSV missing displacement or stress columns")

        if "x" not in fields or "y" not in fields:
            raise ValueError("CSV requires x and y columns")

        if has_disp:
            mapping = {"ux": "ur", "uy": "ut"}
        else:
            mapping = {"sig_x": "sig_r", "sig_y": "sig_t", "tau_xy": "tau_rt"}

        out_fields = [mapping.get(name, name) for name in fields]
        rows = list(reader)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(out_fields)

        for row in rows:
            try:
                x = float(row["x"])
                y = float(row["y"])
            except ValueError:
                raise ValueError("x or y is not numeric")

            dx = x - cx
            dy = y - cy
            r = (dx * dx + dy * dy) ** 0.5
            if r == 0.0:
                c = 1.0
                s = 0.0
            else:
                c = dx / r
                s = dy / r

            ux_val = uy_val = None
            sx_val = sy_val = txy_val = None
            if has_disp:
                try:
                    ux_val = float(row["ux"])
                    uy_val = float(row["uy"])
                except ValueError:
                    raise ValueError("ux or uy is not numeric")

                ur = c * ux_val + s * uy_val
                ut = -s * ux_val + c * uy_val

            if has_stress:
                try:
                    sx_val = float(row["sig_x"])
                    sy_val = float(row["sig_y"])
                    txy_val = float(row["tau_xy"])
                except ValueError:
                    raise ValueError("sig_x, sig_y, or tau_xy is not numeric")

                sig_r = c * c * sx_val + s * s * sy_val + 2.0 * s * c * txy_val
                sig_t = s * s * sx_val + c * c * sy_val - 2.0 * s * c * txy_val
                tau_rt = -s * c * sx_val + s * c * sy_val + (c * c - s * s) * txy_val

            out_row = []
            for name in fields:
                if has_disp and name == "ux":
                    out_row.append(ur)
                elif has_disp and name == "uy":
                    out_row.append(ut)
                elif has_stress and name == "sig_x":
                    out_row.append(sig_r)
                elif has_stress and name == "sig_y":
                    out_row.append(sig_t)
                elif has_stress and name == "tau_xy":
                    out_row.append(tau_rt)
                else:
                    out_row.append(row.get(name, ""))

            writer.writerow(out_row)
