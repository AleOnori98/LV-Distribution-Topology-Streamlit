from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import networkx as nx
import numpy as np
import pandas as pd

from core.contracts import ReinforcementResult, SESSION_SCHEMA_VERSION
from core.powerflow_network import PFScenarioParams, PyPSAPowerFlowRunner


SELECTION_ALL = "all_lines"
SELECTION_OVERLOADED = "overloaded_only"
SELECTION_FEEDER_PATH = "violating_feeder_path"


@dataclass(frozen=True)
class ReinforcementSettings:
    selection_mode: str = SELECTION_ALL
    cost_per_km_per_kva: float = 0.08
    max_upgrade_factor: float = 4.0
    allow_emergency_load_shedding: bool = False
    shedding_penalty_per_mwh: float = 100000.0
    solver_name: Optional[str] = None
    min_len_km: float = 0.005
    sn_mva: float = 0.1
    check_nonsense: bool = True


def _parse_optimize_status(optimize_out: Any) -> tuple[str, Optional[str]]:
    if isinstance(optimize_out, tuple) and len(optimize_out) >= 2:
        return str(optimize_out[0]), str(optimize_out[1])
    if optimize_out is None:
        return "unknown", None
    return str(optimize_out), None


def _collect_candidate_line_ids(
    *,
    selection_mode: str,
    line_results: pd.DataFrame,
    bus_results: pd.DataFrame,
    slack_bus: int,
) -> set[str]:
    out = line_results.copy()
    out["line_id"] = out["line_id"].astype(str)
    out["u"] = pd.to_numeric(out["bus0"], errors="coerce")
    out["v"] = pd.to_numeric(out["bus1"], errors="coerce")
    out = out.dropna(subset=["u", "v"]).copy()
    out["u"] = out["u"].astype(int)
    out["v"] = out["v"].astype(int)

    if selection_mode == SELECTION_ALL:
        return set(out["line_id"].tolist())

    if selection_mode == SELECTION_OVERLOADED:
        loading = pd.to_numeric(out.get("loading_pu"), errors="coerce")
        return set(out.loc[loading > 1.0, "line_id"].astype(str).tolist())

    if selection_mode == SELECTION_FEEDER_PATH:
        if "violates_limits" in bus_results.columns:
            violated = bus_results.loc[bus_results["violates_limits"], "bus"]
            violated_buses = pd.to_numeric(violated, errors="coerce").dropna().astype(int).drop_duplicates().tolist()
        else:
            violated_buses = []

        G = nx.Graph()
        edge_to_line_id: dict[tuple[int, int], str] = {}
        for row in out[["line_id", "u", "v"]].itertuples(index=False):
            line_id, u, v = str(row[0]), int(row[1]), int(row[2])
            G.add_edge(u, v)
            edge_to_line_id[(min(u, v), max(u, v))] = line_id

        selected: set[str] = set()
        if G.number_of_nodes() == 0:
            return selected

        for bus in violated_buses:
            if bus == int(slack_bus):
                continue
            try:
                path = nx.shortest_path(G, source=int(slack_bus), target=int(bus))
            except Exception:
                continue
            for i in range(len(path) - 1):
                a, b = int(path[i]), int(path[i + 1])
                lid = edge_to_line_id.get((min(a, b), max(a, b)))
                if lid is not None:
                    selected.add(str(lid))
        return selected

    raise ValueError(f"Unknown reinforcement selection_mode='{selection_mode}'.")


def _build_upgraded_line_params_df(
    *,
    base_line_results: pd.DataFrame,
    line_params_df: Optional[pd.DataFrame],
    new_s_nom_kva_by_line: dict[str, float],
) -> pd.DataFrame:
    if line_params_df is not None and "line_id" in line_params_df.columns:
        out = line_params_df.copy()
    else:
        req = ["line_id", "r_ohm_per_km", "x_ohm_per_km", "s_nom_kva"]
        missing = [c for c in req if c not in base_line_results.columns]
        if missing:
            raise ValueError(
                "Cannot build upgraded line parameters from baseline PF output. "
                f"Missing columns: {missing}"
            )
        keep = [c for c in ["line_id", "line_type", "r_ohm_per_km", "x_ohm_per_km", "s_nom_kva"] if c in base_line_results.columns]
        out = base_line_results[keep].drop_duplicates(subset=["line_id"]).copy()

    out["line_id"] = out["line_id"].astype(str)
    out["s_nom_kva"] = pd.to_numeric(out["s_nom_kva"], errors="coerce")
    out = out.dropna(subset=["s_nom_kva"]).copy()
    out["s_nom_kva"] = out["line_id"].map(new_s_nom_kva_by_line).fillna(out["s_nom_kva"]).astype(float)
    return out


def run_reinforcement_optimization(
    *,
    runner: PyPSAPowerFlowRunner,
    hour: int,
    pole_load_dict: dict[int, float],
    params: PFScenarioParams,
    line_params_df: Optional[pd.DataFrame],
    settings: ReinforcementSettings,
    pre_summary: Optional[dict[str, Any]] = None,
) -> ReinforcementResult:
    if settings.cost_per_km_per_kva <= 0:
        raise ValueError("cost_per_km_per_kva must be > 0.")
    if settings.max_upgrade_factor < 1.0:
        raise ValueError("max_upgrade_factor must be >= 1.")

    # Baseline rebuild using existing runner to guarantee identical topology and load placement.
    baseline_out = runner.run_snapshot(
        pole_p_kw=pole_load_dict,
        params=params,
        line_params_df=line_params_df,
        debug=True,
        check_nonsense=False,
        min_len_km=float(settings.min_len_km),
        sn_mva=float(settings.sn_mva),
    )
    n = baseline_out.get("network")
    if n is None:
        raise RuntimeError("Could not obtain a PyPSA network object from the baseline run.")

    base_line = baseline_out["line_results"].copy()
    base_bus = baseline_out["bus_results"].copy()

    selected_ids = _collect_candidate_line_ids(
        selection_mode=settings.selection_mode,
        line_results=base_line,
        bus_results=base_bus,
        slack_bus=int(params.slack_pole_id),
    )
    if settings.selection_mode != SELECTION_ALL and not selected_ids:
        selected_ids = _collect_candidate_line_ids(
            selection_mode=SELECTION_OVERLOADED,
            line_results=base_line,
            bus_results=base_bus,
            slack_bus=int(params.slack_pole_id),
        )

    lines_ref = base_line[["line_id", "length_km", "bus0", "bus1", "s_nom_kva"]].copy()
    lines_ref["line_id"] = lines_ref["line_id"].astype(str)
    lines_ref = lines_ref.drop_duplicates(subset=["line_id"]).set_index("line_id")

    n.lines["s_nom"] = pd.to_numeric(n.lines["s_nom"], errors="coerce").fillna(0.0).astype(float)
    n.lines["s_nom_min"] = n.lines["s_nom"].astype(float)
    n.lines["s_nom_max"] = n.lines["s_nom"].astype(float)
    n.lines["capital_cost"] = 0.0
    n.lines["s_nom_extendable"] = False

    for line_id in n.lines.index.astype(str):
        if line_id not in selected_ids:
            continue
        old_s_nom_mva = float(n.lines.at[line_id, "s_nom"])
        n.lines.at[line_id, "s_nom_extendable"] = True
        n.lines.at[line_id, "s_nom_min"] = old_s_nom_mva
        n.lines.at[line_id, "s_nom_max"] = old_s_nom_mva * float(settings.max_upgrade_factor)
        length_km = float(lines_ref.at[line_id, "length_km"]) if line_id in lines_ref.index else 0.0
        n.lines.at[line_id, "capital_cost"] = float(length_km) * float(settings.cost_per_km_per_kva) * 1000.0

    if settings.allow_emergency_load_shedding:
        for load_name, row in n.loads.iterrows():
            bus = str(row["bus"])
            p_set_mw = float(pd.to_numeric(row.get("p_set"), errors="coerce") or 0.0)
            if p_set_mw <= 0:
                continue
            shed_name = f"shedding_{load_name}"
            if shed_name in n.generators.index:
                continue
            n.add(
                "Generator",
                name=shed_name,
                bus=bus,
                p_nom=float(p_set_mw),
                marginal_cost=float(settings.shedding_penalty_per_mwh),
                carrier="load_shedding",
            )

    optimize_kwargs: dict[str, Any] = {}
    if settings.solver_name:
        optimize_kwargs["solver_name"] = str(settings.solver_name)
    optimize_out = n.optimize(**optimize_kwargs)
    opt_status, termination = _parse_optimize_status(optimize_out)

    objective_value = None
    if hasattr(n, "objective"):
        try:
            objective_value = float(n.objective)
        except Exception:
            objective_value = None

    if hasattr(n.optimize, "fix_optimal_dispatch"):
        try:
            n.optimize.fix_optimal_dispatch()
        except Exception:
            pass

    optimize_pf_status = "not_run"
    try:
        n.pf(use_seed=True)
        optimize_pf_status = "ok"
    except Exception as exc:
        optimize_pf_status = f"failed: {repr(exc)}"

    old_s_nom = pd.to_numeric(n.lines["s_nom_min"], errors="coerce").fillna(0.0).astype(float)
    if "s_nom_opt" in n.lines.columns:
        new_s_nom = pd.to_numeric(n.lines["s_nom_opt"], errors="coerce").fillna(old_s_nom).astype(float)
    else:
        new_s_nom = old_s_nom.copy()
    delta_s_nom = (new_s_nom - old_s_nom).clip(lower=0.0)

    rows: list[dict[str, Any]] = []
    for line_id in n.lines.index.astype(str):
        if line_id not in lines_ref.index:
            continue
        delta_mva = float(delta_s_nom.get(line_id, 0.0))
        old_kva = float(old_s_nom.get(line_id, 0.0) * 1000.0)
        new_kva = float(new_s_nom.get(line_id, old_s_nom.get(line_id, 0.0)) * 1000.0)
        length_km = float(lines_ref.at[line_id, "length_km"])
        cost = float(length_km * delta_mva * 1000.0 * settings.cost_per_km_per_kva)
        u = pd.to_numeric(lines_ref.at[line_id, "bus0"], errors="coerce")
        v = pd.to_numeric(lines_ref.at[line_id, "bus1"], errors="coerce")
        rows.append(
            {
                "line_id": line_id,
                "from_bus": int(u) if pd.notna(u) else -1,
                "to_bus": int(v) if pd.notna(v) else -1,
                "length_km": length_km,
                "old_s_nom_kva": old_kva,
                "new_s_nom_kva": new_kva,
                "delta_s_nom_kva": float(max(0.0, new_kva - old_kva)),
                "estimated_cost": cost,
            }
        )

    reinforced_df = pd.DataFrame(rows).sort_values(
        ["delta_s_nom_kva", "line_id"], ascending=[False, True]
    ).reset_index(drop=True)
    reinforced_df = reinforced_df.loc[reinforced_df["delta_s_nom_kva"] > 1e-6].reset_index(drop=True)

    new_s_nom_kva_by_line = {str(k): float(v * 1000.0) for k, v in new_s_nom.to_dict().items()}
    upgraded_line_params_df = _build_upgraded_line_params_df(
        base_line_results=base_line,
        line_params_df=line_params_df,
        new_s_nom_kva_by_line=new_s_nom_kva_by_line,
    )

    post_out = runner.run_snapshot(
        pole_p_kw=pole_load_dict,
        params=params,
        line_params_df=upgraded_line_params_df,
        debug=True,
        check_nonsense=bool(settings.check_nonsense),
        min_len_km=float(settings.min_len_km),
        sn_mva=float(settings.sn_mva),
    )

    total_added_capacity_kva = float(reinforced_df["delta_s_nom_kva"].sum()) if not reinforced_df.empty else 0.0
    total_cost = float(reinforced_df["estimated_cost"].sum()) if not reinforced_df.empty else 0.0

    optimize_debug = {
        "selected_lines_count": int(len(selected_ids)),
        "selected_lines_preview": sorted(list(selected_ids))[:20],
        "optimize_pf_status": optimize_pf_status,
        "optimize_status_raw": optimize_out,
    }

    return ReinforcementResult(
        schema_version=SESSION_SCHEMA_VERSION,
        hour=int(hour),
        settings={
            "selection_mode": settings.selection_mode,
            "cost_per_km_per_kva": float(settings.cost_per_km_per_kva),
            "max_upgrade_factor": float(settings.max_upgrade_factor),
            "allow_emergency_load_shedding": bool(settings.allow_emergency_load_shedding),
            "shedding_penalty_per_mwh": float(settings.shedding_penalty_per_mwh),
            "solver_name": settings.solver_name,
            "min_len_km": float(settings.min_len_km),
            "sn_mva": float(settings.sn_mva),
            "check_nonsense": bool(settings.check_nonsense),
        },
        optimization_status=(opt_status if termination is None else f"{opt_status} ({termination})"),
        objective_value=objective_value,
        pre_summary=dict(pre_summary or baseline_out["summary"]),
        post_summary=dict(post_out["summary"]),
        reinforced_lines=reinforced_df,
        post_bus_results=post_out["bus_results"].copy(),
        post_line_results=post_out["line_results"].copy(),
        total_added_capacity_kva=float(total_added_capacity_kva),
        total_reinforcement_cost=float(total_cost),
        upgraded_line_params_df=upgraded_line_params_df,
        optimize_debug=optimize_debug,
    )
