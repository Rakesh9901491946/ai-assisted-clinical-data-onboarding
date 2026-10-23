from fastapi.testclient import TestClient

from onboarding.api import app


def test_approval_required_then_validated_csv_download(monkeypatch):
    monkeypatch.setattr("onboarding.api.record_upload", lambda *a: None)
    monkeypatch.setattr("onboarding.api.audit", lambda *a: None)
    client = TestClient(app)
    upload = client.post("/uploads", data={"partner_name":"Synthetic"}, files={"file":("tiny.csv", b"patient_id,diagnosis_code,diagnosis_date\nA1,i10,2024-01-01\n", "text/csv")})
    assert upload.status_code == 200
    uid = upload.json()["upload_id"]
    assert client.post(f"/uploads/{uid}/transform").status_code == 403
    mappings = client.post(f"/uploads/{uid}/mappings", params={"use_ai":"false"}).json()["mappings"]
    for item in mappings:
        item["approved"] = True
    approved = client.post(f"/uploads/{uid}/approve", json={"mappings":mappings})
    assert approved.status_code == 200
    transformed = client.post(f"/uploads/{uid}/transform")
    assert transformed.status_code == 200, transformed.text
    csv = client.get(transformed.json()["download_url"])
    assert csv.status_code == 200
    assert "person_source_value,condition_source_value,condition_start_date" in csv.text


def test_partner_alpha_inconsistent_dates_validate_after_approval(monkeypatch):
    monkeypatch.setattr("onboarding.api.record_upload", lambda *a: None)
    monkeypatch.setattr("onboarding.api.audit", lambda *a: None)
    client = TestClient(app)
    content = b"patient_id,diagnosis_code,diagnosis_date\nA1,i10,01/15/2024\nA2,E11.9,2024-02-03\nA3,Z00.00,03-21-2024\n"
    uid = client.post("/uploads", data={"partner_name":"Synthetic"}, files={"file":("mixed.csv", content, "text/csv")}).json()["upload_id"]
    mappings = client.post(f"/uploads/{uid}/mappings", params={"use_ai":"false"}).json()["mappings"]
    for mapping in mappings:
        mapping["approved"] = True
    assert client.post(f"/uploads/{uid}/approve", json={"mappings":mappings}).status_code == 200
    result = client.post(f"/uploads/{uid}/transform").json()
    assert result["validation_errors"] == []
    assert [row["condition_start_date"] for row in result["preview"]] == ["2024-01-15", "2024-02-03", "2024-03-21"]


def test_openrouter_failure_returns_deterministic_fallback(monkeypatch):
    monkeypatch.setattr("onboarding.api.record_upload", lambda *a: None)
    monkeypatch.setattr("onboarding.api.audit", lambda *a: None)
    monkeypatch.setattr("onboarding.api.propose_with_model", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("provider timeout")))
    client = TestClient(app)
    upload = client.post("/uploads", data={"partner_name":"Synthetic"}, files={"file":("tiny.csv", b"patient_id,diagnosis_code,diagnosis_date\nA1,I10,2024-01-01\n", "text/csv")})
    uid = upload.json()["upload_id"]
    response = client.post(f"/uploads/{uid}/mappings", params={"use_ai":"true"})
    assert response.status_code == 200
    assert response.json()["ai_assisted"] is False
    assert len(response.json()["mappings"]) == 3


def test_approved_workflow_sends_transformed_rows_to_project1_staging(monkeypatch):
    monkeypatch.setattr("onboarding.api.record_upload", lambda *a: None)
    monkeypatch.setattr("onboarding.api.audit", lambda *a: None)
    captured = {}
    def capture(frame, partner):
        captured["rows"] = frame.to_dict(orient="records")
        captured["partner"] = partner
        return len(frame)
    monkeypatch.setattr("onboarding.api.write_omop_staging", capture)
    client = TestClient(app)
    upload = client.post("/uploads", data={"partner_name":"Synthetic partner"}, files={"file":("tiny.csv", b"patient_id,diagnosis_code,diagnosis_date\nA1,i10,2024-01-01\n", "text/csv")})
    uid = upload.json()["upload_id"]
    proposals = client.post(f"/uploads/{uid}/mappings", params={"use_ai":"false"}).json()["mappings"]
    for proposal in proposals:
        proposal["approved"] = True
    assert client.post(f"/uploads/{uid}/approve", json={"mappings":proposals}).status_code == 200
    result = client.post(f"/uploads/{uid}/transform", params={"send_to_omop_staging":"true"})
    assert result.status_code == 200
    assert result.json()["omop_staged_rows"] == 1
    assert captured["partner"] == "Synthetic partner"
    assert captured["rows"][0] == {"person_source_value":"A1", "condition_source_value":"I10", "condition_start_date":"2024-01-01"}

