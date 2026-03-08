from __future__ import annotations

from dataclasses import replace
from typing import Any, MutableMapping, Optional, Tuple

from .contracts import (
    ProjectDomain,
    ReinforcementResult,
    SESSION_SCHEMA_VERSION,
    TopologyDomain,
    TopologyResult,
    UIDomain,
    ValidationDomain,
    ValidationInputs,
    ValidationResult,
)


def ensure_session_domains(session_state: MutableMapping[str, Any]) -> None:
    if "project" not in session_state or not isinstance(session_state["project"], dict):
        session_state["project"] = ProjectDomain(version=SESSION_SCHEMA_VERSION)
    else:
        session_state["project"].setdefault("version", SESSION_SCHEMA_VERSION)

    if "topology" not in session_state or not isinstance(session_state["topology"], dict):
        session_state["topology"] = TopologyDomain(
            version=SESSION_SCHEMA_VERSION,
            result=None,
            solve_seconds=None,
        )
    else:
        session_state["topology"].setdefault("version", SESSION_SCHEMA_VERSION)
        session_state["topology"].setdefault("result", None)
        session_state["topology"].setdefault("solve_seconds", None)

    if "validation" not in session_state or not isinstance(session_state["validation"], dict):
        session_state["validation"] = ValidationDomain(
            version=SESSION_SCHEMA_VERSION,
            inputs=None,
            result=None,
            reinforcement=None,
            runner=None,
            topo_fingerprint=None,
        )
    else:
        session_state["validation"].setdefault("version", SESSION_SCHEMA_VERSION)
        session_state["validation"].setdefault("inputs", None)
        session_state["validation"].setdefault("result", None)
        session_state["validation"].setdefault("reinforcement", None)
        session_state["validation"].setdefault("runner", None)
        session_state["validation"].setdefault("topo_fingerprint", None)

    if "ui" not in session_state or not isinstance(session_state["ui"], dict):
        session_state["ui"] = UIDomain(version=SESSION_SCHEMA_VERSION, flags={})
    else:
        session_state["ui"].setdefault("version", SESSION_SCHEMA_VERSION)
        session_state["ui"].setdefault("flags", {})


def set_project_request(
    session_state: MutableMapping[str, Any],
    *,
    topology_request: Optional[dict[str, Any]] = None,
    validation_request: Optional[dict[str, Any]] = None,
) -> None:
    ensure_session_domains(session_state)
    project = session_state["project"]
    if topology_request is not None:
        project["topology_request"] = topology_request
    if validation_request is not None:
        project["validation_request"] = validation_request


def get_topology_result(session_state: MutableMapping[str, Any]) -> Optional[TopologyResult]:
    ensure_session_domains(session_state)
    return session_state["topology"].get("result")


def get_topology_solve_seconds(session_state: MutableMapping[str, Any]) -> Optional[float]:
    ensure_session_domains(session_state)
    return session_state["topology"].get("solve_seconds")


def set_topology_result(
    session_state: MutableMapping[str, Any],
    result: Optional[TopologyResult],
    solve_seconds: Optional[float] = None,
) -> None:
    ensure_session_domains(session_state)
    session_state["topology"]["result"] = result
    session_state["topology"]["solve_seconds"] = solve_seconds


def clear_topology(session_state: MutableMapping[str, Any], *, clear_validation: bool = True) -> None:
    ensure_session_domains(session_state)
    session_state["topology"]["result"] = None
    session_state["topology"]["solve_seconds"] = None
    if clear_validation:
        clear_validation_state(session_state)


def get_validation_inputs(session_state: MutableMapping[str, Any]) -> Optional[ValidationInputs]:
    ensure_session_domains(session_state)
    return session_state["validation"].get("inputs")


def set_validation_inputs(
    session_state: MutableMapping[str, Any],
    inputs: Optional[ValidationInputs],
    *,
    reset_runtime: bool = False,
) -> None:
    ensure_session_domains(session_state)
    session_state["validation"]["inputs"] = inputs
    if inputs is None or reset_runtime:
        session_state["validation"]["result"] = None
        session_state["validation"]["reinforcement"] = None
        session_state["validation"]["runner"] = None
        session_state["validation"]["topo_fingerprint"] = None


def update_validation_inputs(session_state: MutableMapping[str, Any], inputs: ValidationInputs) -> None:
    ensure_session_domains(session_state)
    session_state["validation"]["inputs"] = inputs


def update_validation_load_state(
    session_state: MutableMapping[str, Any],
    *,
    pole_loads_kW,
    selected_hour: int,
    pole_load_dict: dict[int, float],
    scaling_mode: str,
    pmax_ref_kW: Optional[float],
    year_max_pole_kW: float,
) -> ValidationInputs:
    inputs = get_validation_inputs(session_state)
    if inputs is None:
        raise ValueError("Validation inputs are not initialized.")

    updated = replace(
        inputs,
        pole_loads_kW=pole_loads_kW,
        selected_hour=int(selected_hour),
        pole_load_dict=dict(pole_load_dict),
        scaling_mode=str(scaling_mode),
        pmax_ref_kW=None if pmax_ref_kW is None else float(pmax_ref_kW),
        year_max_pole_kW=float(year_max_pole_kW),
    )
    update_validation_inputs(session_state, updated)
    return updated


def update_validation_pf_settings(
    session_state: MutableMapping[str, Any],
    *,
    slack_pole_id: int,
    v_min_pu: float,
    v_max_pu: float,
    pf_load: float,
    v_nom_kv: float,
    v_base_mode: str,
    r_ohm_per_km: float,
    x_ohm_per_km: float,
    s_nom_kva: float,
) -> ValidationInputs:
    inputs = get_validation_inputs(session_state)
    if inputs is None:
        raise ValueError("Validation inputs are not initialized.")

    updated = replace(
        inputs,
        slack_pole_id=int(slack_pole_id),
        v_min_pu=float(v_min_pu),
        v_max_pu=float(v_max_pu),
        pf_load=float(pf_load),
        v_nom_kv=float(v_nom_kv),
        v_base_mode=str(v_base_mode),
        r_ohm_per_km=float(r_ohm_per_km),
        x_ohm_per_km=float(x_ohm_per_km),
        s_nom_kva=float(s_nom_kva),
    )
    update_validation_inputs(session_state, updated)
    return updated


def update_validation_line_params_state(
    session_state: MutableMapping[str, Any],
    *,
    line_params_mode: str,
    default_line_type: Optional[str],
    line_types_df,
    lines_meta_df,
    resolved_line_params_df,
) -> ValidationInputs:
    inputs = get_validation_inputs(session_state)
    if inputs is None:
        raise ValueError("Validation inputs are not initialized.")

    updated = replace(
        inputs,
        line_params_mode=str(line_params_mode),
        default_line_type=None if default_line_type in (None, "") else str(default_line_type),
        line_types_df=line_types_df,
        lines_meta_df=lines_meta_df,
        resolved_line_params_df=resolved_line_params_df,
    )
    update_validation_inputs(session_state, updated)
    return updated


def get_validation_result(session_state: MutableMapping[str, Any]) -> Optional[ValidationResult]:
    ensure_session_domains(session_state)
    return session_state["validation"].get("result")


def set_validation_result(session_state: MutableMapping[str, Any], result: Optional[ValidationResult]) -> None:
    ensure_session_domains(session_state)
    session_state["validation"]["result"] = result
    session_state["validation"]["reinforcement"] = None


def get_validation_reinforcement_result(
    session_state: MutableMapping[str, Any],
) -> Optional[ReinforcementResult]:
    ensure_session_domains(session_state)
    return session_state["validation"].get("reinforcement")


def set_validation_reinforcement_result(
    session_state: MutableMapping[str, Any],
    result: Optional[ReinforcementResult],
) -> None:
    ensure_session_domains(session_state)
    session_state["validation"]["reinforcement"] = result


def get_validation_runner_cache(session_state: MutableMapping[str, Any]) -> tuple[Any, Optional[Tuple[Any, ...]]]:
    ensure_session_domains(session_state)
    validation = session_state["validation"]
    return validation.get("runner"), validation.get("topo_fingerprint")


def set_validation_runner_cache(
    session_state: MutableMapping[str, Any],
    runner: Any,
    topo_fingerprint: Tuple[Any, ...],
) -> None:
    ensure_session_domains(session_state)
    session_state["validation"]["runner"] = runner
    session_state["validation"]["topo_fingerprint"] = topo_fingerprint


def clear_validation_state(session_state: MutableMapping[str, Any]) -> None:
    ensure_session_domains(session_state)
    session_state["validation"]["inputs"] = None
    session_state["validation"]["result"] = None
    session_state["validation"]["reinforcement"] = None
    session_state["validation"]["runner"] = None
    session_state["validation"]["topo_fingerprint"] = None
