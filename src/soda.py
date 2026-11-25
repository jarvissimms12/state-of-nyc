"""Small client for the NYC Open Data (Socrata SODA) API.

Two modes:
  * query()  - one SoQL query, used for server-side GROUP BY aggregates
  * fetch_all() - pages through row-level results with a stable ORDER BY

Retries with backoff because the API returns intermittent 503s under load.
An app token (env var SODA_APP_TOKEN) raises rate limits but is optional.
"""
from __future__ import annotations

import os
import time

import pandas as pd
import requests

from .config import SODA_DOMAIN

PAGE = 50_000


def query(dataset: str, soql: str, retries: int = 5, timeout: int = 300) -> pd.DataFrame:
    """Run one SoQL query and return a DataFrame (all columns as strings)."""
    url = f"{SODA_DOMAIN}/resource/{dataset}.json"
    headers = {}
    if token := os.environ.get("SODA_APP_TOKEN"):
        headers["X-App-Token"] = token
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            resp = requests.get(url, params={"$query": soql}, headers=headers, timeout=timeout)
            if resp.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError(f"{resp.status_code}: {resp.text[:200]}")
            resp.raise_for_status()
            return pd.DataFrame(resp.json())
        except (requests.RequestException, ValueError) as err:
            last_err = err
            time.sleep(min(60, 2 ** attempt * 3))
    raise RuntimeError(f"SODA query failed after {retries} attempts on {dataset}: {last_err}")


def geojson(dataset: str, retries: int = 5) -> dict:
    """Download a dataset's geometry as GeoJSON, with the same retry policy."""
    url = f"{SODA_DOMAIN}/resource/{dataset}.geojson"
    for attempt in range(retries):
        try:
            resp = requests.get(url, params={"$limit": 1000}, timeout=300)
            data = resp.json()
            if resp.ok and "features" in data:
                return data
        except (requests.RequestException, ValueError):
            pass
        time.sleep(min(60, 2 ** attempt * 3))
    raise RuntimeError(f"GeoJSON download failed after {retries} attempts on {dataset}")


def fetch_all(dataset: str, select: str, where: str | None, order: str) -> pd.DataFrame:
    """Page through all rows matching a query. `order` must be unique-ish and stable."""
    frames, offset = [], 0
    clause = f" WHERE {where}" if where else ""
    while True:
        soql = f"SELECT {select}{clause} ORDER BY {order} LIMIT {PAGE} OFFSET {offset}"
        page = query(dataset, soql)
        frames.append(page)
        if len(page) < PAGE:
            break
        offset += PAGE
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
