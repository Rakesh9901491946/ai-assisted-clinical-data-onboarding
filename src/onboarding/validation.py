from __future__ import annotations

import pandas as pd
import re


def validate_output(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    errors = []
    required = {"person_source_value", "condition_source_value", "condition_start_date"}
    missing_columns = required - set(frame.columns)
    for field in sorted(missing_columns):
        errors.append({"row": None, "column": field, "message": "Required target field is missing"})
    if not missing_columns:
        for field in required:
            for idx in frame.index[frame[field].isna() | frame[field].astype("string").str.strip().eq("")]:
                errors.append({"row": int(idx) + 1, "column": field, "message": "Value is required"})
        for idx, value in frame["condition_start_date"].items():
            parsed = pd.to_datetime(value, errors="coerce", format="%Y-%m-%d") if pd.notna(value) else pd.NaT
            if pd.notna(value) and pd.isna(parsed):
                errors.append({"row": int(idx) + 1, "column": "condition_start_date", "message": "Date must use YYYY-MM-DD"})
        for idx, value in frame["condition_source_value"].items():
            if pd.notna(value) and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,63}", str(value)):
                errors.append({"row": int(idx) + 1, "column": "condition_source_value", "message": "Code contains unsupported characters or exceeds 64 characters"})
    # Pandera contract is applied to the normalized schema as an independent check.
    try:
        import pandera.pandas as pa
        schema = pa.DataFrameSchema({
            "person_source_value": pa.Column(str, nullable=False),
            "condition_source_value": pa.Column(str, nullable=False),
            "condition_start_date": pa.Column(str, nullable=False, checks=pa.Check.str_matches(r"^\d{4}-\d{2}-\d{2}$")),
        }, strict=False, coerce=True)
        schema.validate(frame, lazy=True)
    except ImportError:
        pass
    except Exception as exc:
        errors.append({"row": None, "column": "schema", "message": str(exc)[:500]})
    return frame, errors

