from __future__ import annotations

import pandas as pd

from core.line_params import build_line_params_for_edges


def main() -> None:
    edges_df = pd.DataFrame(
        [
            {"line_id": "L0", "u": 1, "v": 2, "length_km": 0.050},
            {"line_id": "L1", "u": 2, "v": 3, "length_km": 0.025},
            {"line_id": "L2", "u": 3, "v": 4, "length_km": 0.010},
        ]
    )
    line_types_df = pd.DataFrame(
        [
            {"line_type": "main", "r_ohm_per_km": 0.50, "x_ohm_per_km": 0.08, "s_nom_kva": 150.0},
            {"line_type": "service", "r_ohm_per_km": 0.90, "x_ohm_per_km": 0.10, "s_nom_kva": 60.0},
        ]
    )
    lines_meta_df = pd.DataFrame(
        [
            {"line_id": "L0", "line_type": "main"},
            {"line_id": "L1", "line_type": "service", "s_nom_kva_override": 80.0},
        ]
    )
    default_params = {"r_ohm_per_km": 0.642, "x_ohm_per_km": 0.083, "s_nom_kva": 100.0}

    for mode in ["global", "catalog", "catalog_overrides"]:
        out = build_line_params_for_edges(
            edges_df,
            mode=mode,
            default_params=default_params,
            default_line_type="main",
            line_types_df=line_types_df,
            lines_meta_df=lines_meta_df,
        )
        print(f"\nMODE: {mode}")
        print(out[["line_id", "line_type", "r_ohm_per_km", "x_ohm_per_km", "s_nom_kva"]].to_string(index=False))


if __name__ == "__main__":
    main()
