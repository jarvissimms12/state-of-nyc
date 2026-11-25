"""Community district boundaries and names.

Boundaries come from DCP's Community Districts layer (71 shapes, including
parks and airports). Coordinates are rounded to 4 decimals (~10 m) and
consecutive duplicate points dropped, which shrinks the file ~10x with no
visible change at city scale.
"""
from __future__ import annotations

import json

import pandas as pd
from src import soda
from src.config import DATASETS, PROCESSED
from src.geo import VALID_CDS, cd_code


def _simplify_ring(ring: list, ndigits: int = 4) -> list:
    out = []
    for x, y in ring:
        pt = [round(x, ndigits), round(y, ndigits)]
        if not out or pt != out[-1]:
            out.append(pt)
    if out[0] != out[-1]:
        out.append(out[0])
    return out


def boundaries() -> dict:
    gj = soda.geojson(DATASETS["cd_boundaries"])
    features = []
    for f in gj["features"]:
        cd = int(float(f["properties"]["boro_cd"]))
        if cd not in VALID_CDS:
            continue
        geom = f["geometry"]
        geom["coordinates"] = [[_simplify_ring(r) for r in poly] for poly in geom["coordinates"]]
        features.append({"type": "Feature", "id": cd, "properties": {"cd": cd}, "geometry": geom})
    out = {"type": "FeatureCollection", "features": features}
    (PROCESSED / "community_districts.geojson").write_text(json.dumps(out, separators=(",", ":")))
    return out


def names() -> pd.DataFrame:
    df = soda.query("xi7c-iiu2", "SELECT borough, cd_number, cd_name LIMIT 100")
    df["cd"] = [cd_code(b, n) for b, n in zip(df["borough"], df["cd_number"])]
    return df.dropna(subset=["cd"]).astype({"cd": int})[["cd", "cd_name"]]
