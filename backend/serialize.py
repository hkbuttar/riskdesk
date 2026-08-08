"""JSON-safe serialization for backend responses.

Every module in this project was built to return real Python objects
(dataclasses, pandas Series/DataFrames, numpy scalars) suited to the
scripts and tests that consume them directly -- not pre-flattened into
JSON-safe dicts, since that would have coupled every module's return type
to "whatever the eventual API needs," a dependency running the wrong
direction. This module is the one place that conversion happens, so
`backend/main.py`'s route handlers stay thin wiring, not re-implementations
of what each module already computed.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import enum
import math

import numpy as np
import pandas as pd


def to_jsonable(obj):
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        return None if math.isnan(obj) or math.isinf(obj) else obj
    if isinstance(obj, (np.floating,)):
        return to_jsonable(float(obj))
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return [to_jsonable(x) for x in obj.tolist()]
    if isinstance(obj, enum.Enum):
        return obj.value
    if isinstance(obj, (dt.datetime, dt.date, pd.Timestamp)):
        return obj.isoformat()
    if isinstance(obj, pd.Series):
        return {to_jsonable(k): to_jsonable(v) for k, v in obj.to_dict().items()}
    if isinstance(obj, pd.Index):
        return [to_jsonable(value) for value in obj.tolist()]
    if isinstance(obj, pd.DataFrame):
        return {
            to_jsonable(idx): {to_jsonable(col): to_jsonable(v) for col, v in row.items()}
            for idx, row in obj.iterrows()
        }
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_jsonable(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, dict):
        return {to_jsonable(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(x) for x in obj]
    return obj
