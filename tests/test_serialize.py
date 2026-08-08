"""Tests for backend/serialize.py's JSON-safe conversion -- the one place
every route handler's real return types (dataclasses, pandas, numpy) get
flattened into JSON. Checked type by type against known inputs, since a
silent mis-serialization (e.g. a NaN slipping through as literal `NaN` in
the JSON body, which most parsers reject) would only surface as a vague
downstream frontend bug otherwise.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import enum
import math

import numpy as np
import pandas as pd
import pytest

from backend.serialize import to_jsonable


def test_primitives_pass_through_unchanged():
    assert to_jsonable(None) is None
    assert to_jsonable(True) is True
    assert to_jsonable(42) == 42
    assert to_jsonable("abc") == "abc"
    assert to_jsonable(1.5) == 1.5


def test_nan_and_inf_become_none_not_invalid_json():
    assert to_jsonable(float("nan")) is None
    assert to_jsonable(float("inf")) is None
    assert to_jsonable(float("-inf")) is None


def test_numpy_scalars_convert_to_native_python():
    assert isinstance(to_jsonable(np.float64(1.5)), float)
    assert isinstance(to_jsonable(np.int64(5)), int)
    assert to_jsonable(np.float64(1.5)) == 1.5
    assert to_jsonable(np.nan) is None  # numpy NaN also caught


def test_numpy_array_converts_to_list():
    arr = np.array([1.0, 2.0, 3.0])
    result = to_jsonable(arr)
    assert result == [1.0, 2.0, 3.0]
    assert isinstance(result, list)


def test_enum_converts_to_its_value():
    class Color(enum.Enum):
        RED = "red"

    assert to_jsonable(Color.RED) == "red"


def test_dates_convert_to_isoformat_strings():
    d = dt.date(2026, 8, 7)
    assert to_jsonable(d) == "2026-08-07"
    ts = pd.Timestamp("2026-08-07T12:00:00")
    assert to_jsonable(ts) == ts.isoformat()


def test_pandas_series_converts_to_dict():
    s = pd.Series({"a": 1.0, "b": 2.0})
    assert to_jsonable(s) == {"a": 1.0, "b": 2.0}


def test_pandas_index_converts_to_list():
    index = pd.DatetimeIndex(["2026-08-06", "2026-08-07"])
    assert to_jsonable(index) == ["2026-08-06T00:00:00", "2026-08-07T00:00:00"]


def test_pandas_dataframe_converts_to_nested_dict():
    df = pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]}, index=["r1", "r2"])
    result = to_jsonable(df)
    assert result == {"r1": {"x": 1.0, "y": 3.0}, "r2": {"x": 2.0, "y": 4.0}}


def test_dataclass_converts_to_dict_of_fields():
    @dataclasses.dataclass
    class Point:
        x: float
        y: float

    assert to_jsonable(Point(1.0, 2.0)) == {"x": 1.0, "y": 2.0}


def test_nested_structures_convert_recursively():
    @dataclasses.dataclass
    class Inner:
        value: float

    data = {"items": [Inner(np.float64(1.0)), Inner(float("nan"))], "series": pd.Series({"a": 1})}
    result = to_jsonable(data)
    assert result == {"items": [{"value": 1.0}, {"value": None}], "series": {"a": 1}}


def test_result_is_actually_json_serializable():
    import json

    @dataclasses.dataclass
    class Result:
        var: float
        notes: str

    payload = {"result": Result(np.float64(7594.16), "n=500"), "nan_value": float("nan")}
    # Must not raise -- this is the real contract: every endpoint's output
    # has to survive an actual json.dumps, not just look plausible.
    json.dumps(to_jsonable(payload))
