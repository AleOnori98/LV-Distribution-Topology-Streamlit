from __future__ import annotations

import unittest

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from core.powerflow_network import PFScenarioParams, PFTopologyBundle, PyPSAPowerFlowRunner
from core.powerflow_reinforcement import (
    ReinforcementSettings,
    run_reinforcement_optimization,
)


def _build_runner() -> PyPSAPowerFlowRunner:
    nodes = gpd.GeoDataFrame(
        {
            "pole_id": [1, 2, 3],
            "geometry": [
                Point(12.0000, 45.0000),
                Point(12.0010, 45.0000),
                Point(12.0020, 45.0000),
            ],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )
    topo = PFTopologyBundle(
        gdf_nodes_4326=nodes,
        pole_id_col="pole_id",
        mst_edges_pole_ids=[(1, 2), (2, 3)],
    )
    return PyPSAPowerFlowRunner(topo)


class ReinforcementTests(unittest.TestCase):
    def test_reinforcement_adds_capacity_and_keeps_topology(self) -> None:
        runner = _build_runner()
        params = PFScenarioParams(
            slack_pole_id=1,
            v_min_pu=0.85,
            v_max_pu=1.10,
            pf_load=0.95,
            v_nom_kv=0.4,
            r_ohm_per_km=0.642,
            x_ohm_per_km=0.083,
            s_nom_kva=30.0,
            load_scale=1.0,
        )
        line_params = pd.DataFrame(
            [
                {"line_id": "L0", "r_ohm_per_km": 0.642, "x_ohm_per_km": 0.083, "s_nom_kva": 10.0},
                {"line_id": "L1", "r_ohm_per_km": 0.642, "x_ohm_per_km": 0.083, "s_nom_kva": 10.0},
            ]
        )
        loads = {3: 25.0}

        pre = runner.run_snapshot(
            pole_p_kw=loads,
            params=params,
            line_params_df=line_params,
            debug=True,
            check_nonsense=True,
            min_len_km=0.0,
            sn_mva=0.1,
        )
        out = run_reinforcement_optimization(
            runner=runner,
            hour=0,
            pole_load_dict=loads,
            params=params,
            line_params_df=line_params,
            settings=ReinforcementSettings(
                selection_mode="all_lines",
                cost_per_km_per_kva=0.1,
                max_upgrade_factor=20.0,
                min_len_km=0.0,
                sn_mva=0.1,
            ),
            pre_summary=pre["summary"],
        )

        pre_edges = {
            (int(row["bus0"]), int(row["bus1"]))
            for row in pre["line_results"].to_dict(orient="records")
        }
        post_edges = {
            (int(row["bus0"]), int(row["bus1"]))
            for row in out.post_line_results.to_dict(orient="records")
        }
        self.assertEqual(pre_edges, post_edges)
        self.assertGreater(out.total_added_capacity_kva, 0.0)
        self.assertGreaterEqual(out.total_reinforcement_cost, 0.0)

        # Loads remain fixed at same buses.
        self.assertEqual(set(loads.keys()), {3})

    def test_no_need_case_returns_zero_upgrade(self) -> None:
        runner = _build_runner()
        params = PFScenarioParams(
            slack_pole_id=1,
            v_min_pu=0.85,
            v_max_pu=1.10,
            pf_load=0.95,
            v_nom_kv=0.4,
            r_ohm_per_km=0.642,
            x_ohm_per_km=0.083,
            s_nom_kva=400.0,
            load_scale=1.0,
        )
        line_params = pd.DataFrame(
            [
                {"line_id": "L0", "r_ohm_per_km": 0.642, "x_ohm_per_km": 0.083, "s_nom_kva": 400.0},
                {"line_id": "L1", "r_ohm_per_km": 0.642, "x_ohm_per_km": 0.083, "s_nom_kva": 400.0},
            ]
        )
        loads = {3: 10.0}

        out = run_reinforcement_optimization(
            runner=runner,
            hour=0,
            pole_load_dict=loads,
            params=params,
            line_params_df=line_params,
            settings=ReinforcementSettings(
                selection_mode="overloaded_only",
                cost_per_km_per_kva=0.1,
                max_upgrade_factor=5.0,
                min_len_km=0.0,
                sn_mva=0.1,
            ),
        )

        self.assertAlmostEqual(out.total_added_capacity_kva, 0.0, places=9)
        self.assertAlmostEqual(out.total_reinforcement_cost, 0.0, places=9)
        self.assertTrue(out.reinforced_lines.empty)


if __name__ == "__main__":
    unittest.main()
