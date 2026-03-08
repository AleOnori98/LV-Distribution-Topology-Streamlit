from __future__ import annotations

import io
from typing import Any, Literal

import pandas as pd


LINE_TYPES_REQUIRED = ["line_type", "r_ohm_per_km", "x_ohm_per_km", "s_nom_kva"]
LINES_META_OPTIONAL = [
    "line_type",
    "r_ohm_per_km_override",
    "x_ohm_per_km_override",
    "s_nom_kva_override",
]


def _read_csv_any(file_or_bytes: Any) -> pd.DataFrame:
    if file_or_bytes is None:
        raise ValueError("No CSV file provided.")
    if isinstance(file_or_bytes, pd.DataFrame):
        return file_or_bytes.copy()
    if isinstance(file_or_bytes, (bytes, bytearray)):
        return pd.read_csv(io.BytesIO(file_or_bytes))
    if hasattr(file_or_bytes, "getvalue"):
        return pd.read_csv(io.BytesIO(file_or_bytes.getvalue()))
    return pd.read_csv(file_or_bytes)


def validate_line_types(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        raise ValueError("line_types.csv is empty.")

    missing = [c for c in LINE_TYPES_REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(
            "line_types.csv is missing required columns: "
            f"{missing}. Expected: {LINE_TYPES_REQUIRED}."
        )

    out = df.copy()
    out["line_type"] = out["line_type"].astype(str).str.strip()
    if (out["line_type"] == "").any():
        raise ValueError("line_types.csv contains blank line_type values.")
    if out["line_type"].duplicated().any():
        dups = out.loc[out["line_type"].duplicated(), "line_type"].head(10).tolist()
        raise ValueError(f"line_types.csv contains duplicated line_type values. Examples: {dups}")

    for col in ["r_ohm_per_km", "x_ohm_per_km", "s_nom_kva"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
        if out[col].isna().any():
            bad = out.loc[out[col].isna(), "line_type"].head(10).tolist()
            raise ValueError(f"line_types.csv has non-numeric values in '{col}' for line types: {bad}")

    if (out["r_ohm_per_km"] <= 0).any():
        raise ValueError("line_types.csv requires r_ohm_per_km > 0 for every line type.")
    if (out["x_ohm_per_km"] < 0).any():
        raise ValueError("line_types.csv requires x_ohm_per_km >= 0 for every line type.")
    if (out["s_nom_kva"] <= 0).any():
        raise ValueError("line_types.csv requires s_nom_kva > 0 for every line type.")

    return out.reset_index(drop=True)


def validate_lines_metadata(df: pd.DataFrame) -> pd.DataFrame:
    if df is None:
        raise ValueError("lines_metadata.csv is missing.")
    if df.empty:
        return pd.DataFrame(columns=["line_id"] + LINES_META_OPTIONAL)
    if "line_id" not in df.columns:
        raise ValueError("lines_metadata.csv must contain a 'line_id' column.")

    out = df.copy()
    out["line_id"] = out["line_id"].astype(str).str.strip()
    if (out["line_id"] == "").any():
        raise ValueError("lines_metadata.csv contains blank line_id values.")
    if out["line_id"].duplicated().any():
        dups = out.loc[out["line_id"].duplicated(), "line_id"].head(10).tolist()
        raise ValueError(f"lines_metadata.csv contains duplicated line_id values. Examples: {dups}")

    if "line_type" in out.columns:
        out["line_type"] = out["line_type"].astype(str).str.strip()
        out.loc[out["line_type"] == "", "line_type"] = pd.NA

    for col in ["r_ohm_per_km_override", "x_ohm_per_km_override", "s_nom_kva_override"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    if "r_ohm_per_km_override" in out.columns and (out["r_ohm_per_km_override"].dropna() <= 0).any():
        raise ValueError("lines_metadata.csv requires r_ohm_per_km_override > 0 when provided.")
    if "x_ohm_per_km_override" in out.columns and (out["x_ohm_per_km_override"].dropna() < 0).any():
        raise ValueError("lines_metadata.csv requires x_ohm_per_km_override >= 0 when provided.")
    if "s_nom_kva_override" in out.columns and (out["s_nom_kva_override"].dropna() <= 0).any():
        raise ValueError("lines_metadata.csv requires s_nom_kva_override > 0 when provided.")

    cols = ["line_id"] + [c for c in LINES_META_OPTIONAL if c in out.columns]
    return out[cols].reset_index(drop=True)


def read_line_types_csv(file_or_bytes: Any) -> pd.DataFrame:
    return validate_line_types(_read_csv_any(file_or_bytes))


def read_lines_metadata_csv(file_or_bytes: Any) -> pd.DataFrame:
    return validate_lines_metadata(_read_csv_any(file_or_bytes))


def build_line_params_for_edges(
    edges_df: pd.DataFrame,
    *,
    mode: Literal["global", "catalog", "catalog_overrides"],
    default_params: dict[str, float],
    default_line_type: str | None,
    line_types_df: pd.DataFrame | None,
    lines_meta_df: pd.DataFrame | None,
) -> pd.DataFrame:
    if edges_df is None or edges_df.empty:
        raise ValueError("Cannot build line parameters: edges table is empty.")
    if "line_id" not in edges_df.columns:
        raise ValueError(
            "Edges are missing 'line_id'. Catalog-based line parameters require stable line IDs. "
            "Use exported edges with line_id or rerun topology/validation after updating the app."
        )

    out = edges_df.copy()
    out["line_id"] = out["line_id"].astype(str).str.strip()
    if out["line_id"].duplicated().any():
        dups = out.loc[out["line_id"].duplicated(), "line_id"].head(10).tolist()
        raise ValueError(f"Edges contain duplicated line_id values. Examples: {dups}")

    req_defaults = ["r_ohm_per_km", "x_ohm_per_km", "s_nom_kva"]
    missing_defaults = [c for c in req_defaults if c not in default_params]
    if missing_defaults:
        raise ValueError(f"default_params is missing required keys: {missing_defaults}")
    r_default = float(default_params["r_ohm_per_km"])
    x_default = float(default_params["x_ohm_per_km"])
    s_default = float(default_params["s_nom_kva"])
    if r_default <= 0 or x_default < 0 or s_default <= 0:
        raise ValueError("Global default line parameters must satisfy r>0, x>=0, s_nom>0.")

    if mode == "global":
        out["line_type"] = default_line_type if default_line_type else "global_default"
        out["r_ohm_per_km"] = r_default
        out["x_ohm_per_km"] = x_default
        out["s_nom_kva"] = s_default
        return out

    if line_types_df is None:
        raise ValueError("line_types.csv is required for catalog-based line parameter modes.")

    types_df = validate_line_types(line_types_df)
    if default_line_type is None or str(default_line_type).strip() == "":
        raise ValueError("Select a default_line_type for catalog-based line parameter modes.")
    default_line_type = str(default_line_type).strip()
    if default_line_type not in set(types_df["line_type"].tolist()):
        raise ValueError(
            f"default_line_type='{default_line_type}' is not present in line_types.csv."
        )

    if lines_meta_df is None:
        meta_df = pd.DataFrame({"line_id": out["line_id"].tolist()})
    else:
        meta_df = validate_lines_metadata(lines_meta_df)

    extra_ids = sorted(set(meta_df["line_id"].tolist()) - set(out["line_id"].tolist()))
    if extra_ids:
        raise ValueError(
            "lines_metadata.csv contains line_id values that are not present in the current edges table. "
            f"Examples (up to 20): {extra_ids[:20]}"
        )

    out = out.merge(meta_df, on="line_id", how="left")
    out["line_type"] = out.get("line_type")
    out["line_type"] = out["line_type"].fillna(default_line_type).astype(str).str.strip()

    missing_types = sorted(set(out["line_type"].tolist()) - set(types_df["line_type"].tolist()))
    if missing_types:
        raise ValueError(
            "lines_metadata.csv references line_type values not found in line_types.csv. "
            f"Missing types: {missing_types[:20]}"
        )

    type_lookup = types_df.set_index("line_type")[
        ["r_ohm_per_km", "x_ohm_per_km", "s_nom_kva"]
    ]
    out = out.join(type_lookup, on="line_type", rsuffix="_catalog")

    if mode == "catalog_overrides":
        if "r_ohm_per_km_override" in out.columns:
            out["r_ohm_per_km"] = out["r_ohm_per_km_override"].combine_first(out["r_ohm_per_km"])
        if "x_ohm_per_km_override" in out.columns:
            out["x_ohm_per_km"] = out["x_ohm_per_km_override"].combine_first(out["x_ohm_per_km"])
        if "s_nom_kva_override" in out.columns:
            out["s_nom_kva"] = out["s_nom_kva_override"].combine_first(out["s_nom_kva"])

    for col in ["r_ohm_per_km", "x_ohm_per_km", "s_nom_kva"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
        if out[col].isna().any():
            bad = out.loc[out[col].isna(), "line_id"].head(10).tolist()
            raise ValueError(f"Final line parameter table has missing '{col}' for line_id values: {bad}")

    if (out["r_ohm_per_km"] <= 0).any() or (out["x_ohm_per_km"] < 0).any() or (out["s_nom_kva"] <= 0).any():
        raise ValueError("Final line parameter table must satisfy r>0, x>=0, s_nom>0 for every line.")

    return out
