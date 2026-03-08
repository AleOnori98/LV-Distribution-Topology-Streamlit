from __future__ import annotations

import unittest

import pandas as pd

from core.line_params import build_line_params_for_edges


class LineParamsMergeTests(unittest.TestCase):
    def test_catalog_overrides_and_fallback(self) -> None:
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

        out = build_line_params_for_edges(
            edges_df,
            mode="catalog_overrides",
            default_params={"r_ohm_per_km": 0.642, "x_ohm_per_km": 0.083, "s_nom_kva": 100.0},
            default_line_type="main",
            line_types_df=line_types_df,
            lines_meta_df=lines_meta_df,
        )

        self.assertEqual(out.loc[out["line_id"] == "L0", "line_type"].iloc[0], "main")
        self.assertAlmostEqual(out.loc[out["line_id"] == "L0", "r_ohm_per_km"].iloc[0], 0.50)
        self.assertEqual(out.loc[out["line_id"] == "L1", "line_type"].iloc[0], "service")
        self.assertAlmostEqual(out.loc[out["line_id"] == "L1", "s_nom_kva"].iloc[0], 80.0)
        self.assertEqual(out.loc[out["line_id"] == "L2", "line_type"].iloc[0], "main")
        self.assertAlmostEqual(out.loc[out["line_id"] == "L2", "s_nom_kva"].iloc[0], 150.0)


if __name__ == "__main__":
    unittest.main()
