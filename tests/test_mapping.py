import json
from types import SimpleNamespace

import pandas as pd
import pytest

from onboarding.mapping import MappingProposal, deterministic_proposals, execute_approved, propose_with_model
from onboarding.profiling import profile_csv
from onboarding.validation import validate_output


def test_profile_three_partner_formats():
    frame, profile = profile_csv(b"Member ID,ICD10,Date\nA1,I10,20240116\nA1,I10,20240116\n")
    assert profile["rows"] == 2
    assert profile["duplicate_rows"] == 1
    assert profile["columns"]["ICD10"]["missing_count"] == 0
    assert profile["columns"]["Date"]["dtype"] == "date-like"
    assert "YYYYMMDD" in profile["columns"]["Date"]["date_formats"]
    assert profile["columns"]["ICD10"]["code_format_invalid_count"] == 0
    _, partner_profile = profile_csv(b"patient_id,diagnosis_code,diagnosis_date,clinic_note\nA3,Z00.00,03-21-2024,first\nA3,Z00.00,03-21-2024,duplicate\n")
    assert partner_profile["duplicate_rows"] == 0
    assert partner_profile["duplicate_business_key_rows"] == 1


def test_mixed_dates_parse_without_losing_valid_rows():
    frame = pd.DataFrame({"diagnosis_date": ["01/15/2024", "2024-02-03", "03-21-2024"]})
    proposal = MappingProposal("diagnosis_date", "condition_start_date", "parse_date", "test", 1.0, True)
    result = execute_approved(frame, [proposal])
    assert result["condition_start_date"].tolist() == ["2024-01-15", "2024-02-03", "2024-03-21"]


def test_deterministic_key_maps_and_approval_gate():
    frame = pd.DataFrame({"patient_id": [" A1 "], "diagnosis_code": ["i10"], "diagnosis_date": ["01/15/2024"]})
    proposals = deterministic_proposals(list(frame.columns))
    with pytest.raises(ValueError, match="approved"):
        execute_approved(frame, proposals)
    for proposal in proposals:
        proposal.approved = True
    result = execute_approved(frame, proposals)
    result, errors = validate_output(result)
    assert errors == []
    assert result.iloc[0].to_dict() == {"person_source_value": "A1", "condition_source_value": "I10", "condition_start_date": "2024-01-15"}


def test_pandera_business_rules_catch_missing_date():
    frame = pd.DataFrame({"person_source_value": ["A1"], "condition_source_value": ["I10"], "condition_start_date": [None]})
    _, errors = validate_output(frame)
    assert any(error["column"] == "condition_start_date" for error in errors)


def test_code_lexical_validation_rejects_bad_characters_without_claiming_vocab_match():
    frame = pd.DataFrame({"person_source_value": ["A1"], "condition_source_value": ["I10!"], "condition_start_date": ["2024-01-01"]})
    _, errors = validate_output(frame)
    assert any(error["column"] == "condition_source_value" for error in errors)


def test_model_mock_gets_no_patient_values():
    captured = {}
    def client(**kwargs):
        captured["body"] = kwargs["messages"][1]["content"]
        return json.dumps([{
            "source_column":"diagnosis_code","target_field":"condition_source_value",
            "transformation":"normalize_code","explanation":"Diagnosis code", "confidence":0.99}])
    profile = {"columns":{"diagnosis_code":{"dtype":"string","missing_percent":0}}}
    result = propose_with_model(["diagnosis_code"], profile, client=client)
    assert result[0].confidence == 0.99
    assert "SYN-A001" not in captured["body"]
    assert "row values" in captured["body"]

