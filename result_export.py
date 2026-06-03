# -*- coding: utf-8 -*-
"""
Shareable result export.

Collects every block of data that the pipeline prints to the console and
writes it to a single JSON file (default: result.json) so it can be shared
or diffed without re-running anything.

Usage:
    from result_export import ResultBuilder

    rb = ResultBuilder(symbol="BTCUSD", interval="15m", days=60)
    rb.add("label_stats", symbol_label_stats(...))
    rb.add("backtest_summary", stats)
    rb.add_df("condition_breakdown", bd)
    rb.save("result.json")
"""
from __future__ import annotations

import json
import math
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd


# ── JSON sanitising ───────────────────────────────────────────────────────────

def make_json_safe(obj: Any) -> Any:
    """Recursively convert numpy / pandas / datetime objects into plain
    JSON-serialisable Python types. Non-finite floats (inf / nan) become None."""
    if obj is None or isinstance(obj, (bool, str)):
        return obj

    # numpy scalars → python scalars
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        obj = float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)

    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, int):
        return obj

    if isinstance(obj, (pd.Timestamp, datetime, date)):
        return obj.isoformat()

    if isinstance(obj, np.ndarray):
        return [make_json_safe(x) for x in obj.tolist()]

    if isinstance(obj, pd.DataFrame):
        return [make_json_safe(rec) for rec in obj.to_dict(orient="records")]

    if isinstance(obj, pd.Series):
        return make_json_safe(obj.to_dict())

    if isinstance(obj, dict):
        return {str(k): make_json_safe(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple, set)):
        return [make_json_safe(x) for x in obj]

    # Fallback — best-effort string
    return str(obj)


# ── Builder ───────────────────────────────────────────────────────────────────

class ResultBuilder:
    """Accumulates named sections, then dumps them to one JSON file."""

    def __init__(self, **meta: Any) -> None:
        # Note: generated_at is intentionally left to the caller to pass in
        # (the pipeline can stamp it) to keep this module deterministic.
        self._payload: dict[str, Any] = {"meta": dict(meta)}

    def add(self, section: str, data: Any) -> "ResultBuilder":
        self._payload[section] = make_json_safe(data)
        return self

    def add_df(self, section: str, df: pd.DataFrame) -> "ResultBuilder":
        """Store a DataFrame as a list of row-records."""
        if df is None or df.empty:
            self._payload[section] = []
        else:
            self._payload[section] = make_json_safe(df)
        return self

    def set_meta(self, **meta: Any) -> "ResultBuilder":
        self._payload["meta"].update(make_json_safe(meta))
        return self

    @property
    def payload(self) -> dict[str, Any]:
        return self._payload

    def save(self, path: str = "result.json") -> str:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self._payload, fh, indent=2, ensure_ascii=False)
        return path
