"""PII-minimizing AI mapping proposals and deterministic allowlisted transforms."""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import asdict, dataclass
from typing import Any
from urllib.request import Request, urlopen

import pandas as pd

TARGETS = {"person_source_value", "condition_source_value", "condition_start_date"}
TRANSFORMS = {"identity", "parse_date", "normalize_code", "uppercase"}


@dataclass
class MappingProposal:
    source_column: str
    target_field: str
    transformation: str
    explanation: str
    confidence: float
    approved: bool = False


def deterministic_proposals(columns: list[str]) -> list[MappingProposal]:
    rules = {
        "person_source_value": ("person_source_value", "identity"),
        "patient_id": ("person_source_value", "identity"),
        "patientid": ("person_source_value", "identity"),
        "member_id": ("person_source_value", "identity"),
        "condition_source_value": ("condition_source_value", "normalize_code"),
        "diagnosis_code": ("condition_source_value", "normalize_code"),
        "diagnosiscode": ("condition_source_value", "normalize_code"),
        "icd10": ("condition_source_value", "normalize_code"),
        "code": ("condition_source_value", "normalize_code"),
        "condition_start_date": ("condition_start_date", "parse_date"),
        "diagnosis_date": ("condition_start_date", "parse_date"),
        "diagnosisdate": ("condition_start_date", "parse_date"),
        "date": ("condition_start_date", "parse_date"),
        "onset": ("condition_start_date", "parse_date"),
    }
    proposals = []
    for col in columns:
        key = re.sub(r"[^a-z0-9]", "", col.lower())
        target_transform = rules.get(key)
        if target_transform:
            target, transform = target_transform
            proposals.append(MappingProposal(col, target, transform, "Matched by a deterministic header rule.", 0.94))
    return proposals


def propose_with_model(columns: list[str], profile: dict[str, Any], *, api_key: str | None = None,
                       model: str | None = None, client: Any = None, retries: int = 1) -> list[MappingProposal]:
    """Send only header names and aggregate type/missingness metadata, never row values."""
    api_key = api_key or os.getenv("OPENROUTER_API_KEY")
    if client is None and not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not configured")
    column_profile = profile.get("columns", profile)
    safe_profile = {name: {"dtype": column_profile[name]["dtype"], "missing_percent": column_profile[name]["missing_percent"]}
                    for name in columns if name in column_profile}
    prompt = {
        "task": "Map source headers to target fields. Suggest only valid targets and allowlisted transforms.",
        "targets": sorted(TARGETS), "transforms": sorted(TRANSFORMS),
        "source_columns": columns, "aggregate_profile": safe_profile,
        "output_schema": [{"source_column":"string","target_field":"string","transformation":"string","explanation":"string","confidence":"number 0..1"}],
        "privacy": "Never send patient row values, names, or identifiers. No patient row values are provided.",
    }
    last_error = None
    for attempt in range(retries + 1):
        try:
            request_body = {"model": model or os.getenv("OPENROUTER_MODEL", "openrouter/free"),
                "max_tokens": 1200, "temperature": 0,
                "messages": [{"role":"system", "content":"Return only a JSON array matching the requested schema. Do not return executable code."},
                             {"role":"user", "content":json.dumps(prompt)}]}
            if client is None:
                req = Request("https://openrouter.ai/api/v1/chat/completions",
                    data=json.dumps(request_body).encode(),
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json",
                             "HTTP-Referer": "http://localhost", "X-Title": "Clinical Data Onboarding"}, method="POST")
                with urlopen(req, timeout=20) as response:
                    response_body = json.loads(response.read())
                text = response_body["choices"][0]["message"]["content"]
            else:
                text = client(**request_body)
            if not isinstance(text, str):
                raise ValueError("Model response content must be text")
            if text.startswith("```json"):
                text = text[7:].split("```", 1)[0]
            parsed = json.loads(text)
            return [validate_proposal(item, columns) for item in parsed]
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(0.25 * (attempt + 1))
    raise RuntimeError(f"OpenRouter mapping request failed: {last_error}") from last_error


def validate_proposal(item: dict, columns: list[str]) -> MappingProposal:
    proposal = MappingProposal(**{k: item[k] for k in ("source_column", "target_field", "transformation", "explanation", "confidence")})
    if proposal.source_column not in columns or proposal.target_field not in TARGETS or proposal.transformation not in TRANSFORMS:
        raise ValueError("Mapping contains an unknown source column, target, or transformation")
    if not isinstance(proposal.confidence, (int, float)) or not 0 <= proposal.confidence <= 1:
        raise ValueError("Mapping confidence must be between 0 and 1")
    return proposal


def execute_approved(frame: pd.DataFrame, proposals: list[MappingProposal]) -> pd.DataFrame:
    approved = [p for p in proposals if p.approved]
    if not approved:
        raise ValueError("At least one mapping must be approved before transformation")
    result = pd.DataFrame(index=frame.index)
    for p in approved:
        validate_proposal(asdict(p), list(frame.columns))
        series = frame[p.source_column]
        if p.transformation == "identity":
            value = series.astype("string").str.strip()
        elif p.transformation == "parse_date":
            value = pd.to_datetime(series, errors="coerce", utc=True, format="mixed", dayfirst=False).dt.strftime("%Y-%m-%d")
        elif p.transformation == "normalize_code":
            value = series.astype("string").str.strip().str.upper().str.replace(r"[^A-Z0-9.-]", "", regex=True)
        elif p.transformation == "uppercase":
            value = series.astype("string").str.strip().str.upper()
        else:
            raise ValueError("Unsupported transformation")
        if p.target_field in result:
            raise ValueError(f"More than one approved mapping targets {p.target_field}")
        result[p.target_field] = value
    return result

