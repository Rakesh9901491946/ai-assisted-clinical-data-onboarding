from __future__ import annotations

import pandas as pd
import re


def _infer_type(values: pd.Series, name: str) -> str:
    nonempty = values.dropna().astype(str)
    if not len(nonempty):
        return "empty"
    lowered = nonempty.str.lower()
    if lowered.isin({"true", "false", "yes", "no", "0", "1"}).all():
        return "boolean-like"
    parsed = pd.to_datetime(nonempty, errors="coerce", format="mixed", dayfirst=False)
    if (re.search(r"date|time|onset|effective|authored", name, re.I) and parsed.notna().mean() >= 0.8) or (
            nonempty.str.fullmatch(r"\d{8}").all() and parsed.notna().mean() >= 0.8):
        return "date-like"
    numeric = pd.to_numeric(nonempty, errors="coerce")
    if numeric.notna().all():
        return "integer-like" if numeric.mod(1).eq(0).all() else "numeric"
    if parsed.notna().mean() >= 0.8:
        return "date-like"
    if re.search(r"code|icd|snomed|loinc|rxnorm", name, re.I):
        return "clinical-code-like"
    return "string"


def _date_formats(values: pd.Series) -> list[str]:
    patterns = [
        (r"^\d{4}-\d{2}-\d{2}$", "YYYY-MM-DD"),
        (r"^\d{4}/\d{2}/\d{2}$", "YYYY/MM/DD"),
        (r"^\d{8}$", "YYYYMMDD"),
        (r"^\d{2}/\d{2}/\d{4}$", "MM/DD/YYYY or DD/MM/YYYY"),
        (r"^\d{2}-\d{2}-\d{4}$", "MM-DD-YYYY or DD-MM-YYYY"),
    ]
    found = set()
    for value in values.dropna().astype(str):
        for pattern, label in patterns:
            if re.fullmatch(pattern, value.strip()):
                found.add(label)
    return sorted(found)


def profile_csv(data: bytes) -> tuple[pd.DataFrame, dict]:
    frame = pd.read_csv(pd.io.common.BytesIO(data), dtype="string", keep_default_na=False)
    details = {}
    for column in frame.columns:
        values = frame[column].replace("", pd.NA)
        parsed_dates = pd.to_datetime(values.dropna(), errors="coerce", format="mixed", dayfirst=False)
        details[column] = {"dtype": _infer_type(values, column), "missing_count": int(values.isna().sum()),
            "missing_percent": round(float(values.isna().mean() * 100), 2),
            "distinct_values": int(values.nunique()),
            "date_parseable_percent": round(float(parsed_dates.notna().mean() * 100), 2) if len(parsed_dates) else 0.0,
            "date_formats": _date_formats(values)}
        if re.search(r"code|icd|snomed|loinc|rxnorm", column, re.I):
            code_values = values.dropna().astype(str)
            lexical_ok = code_values.str.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,63}")
            details[column]["code_format_valid_count"] = int(lexical_ok.sum())
            details[column]["code_format_invalid_count"] = int((~lexical_ok).sum())
            details[column]["validation_scope"] = "Lexical format only; no terminology release configured"
    normalized_names = {column: re.sub(r"[^a-z0-9]", "", column.lower()) for column in frame.columns}
    key_columns = []
    for column, name in normalized_names.items():
        if any(token in name for token in ("patientid", "memberid", "personsourcevalue", "patientsourcevalue")):
            key_columns.append(column)
        elif any(token in name for token in ("diagnosiscode", "conditioncode", "conditionvalue", "icd10", "snomed", "loinc", "rxnorm")):
            key_columns.append(column)
        elif any(token in name for token in ("diagnosisdate", "conditionstartdate", "onsetdate")):
            key_columns.append(column)
    return frame, {"rows": len(frame), "columns": details,
        "duplicate_rows": int(frame.duplicated().sum()),
        "duplicate_business_key_rows": int(frame.duplicated(subset=key_columns).sum()) if key_columns else 0,
        "duplicate_key_columns": key_columns,
        "column_names": list(frame.columns)}

