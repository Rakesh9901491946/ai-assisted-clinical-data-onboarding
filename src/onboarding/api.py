from __future__ import annotations

import json
import os
import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response

from .mapping import MappingProposal, deterministic_proposals, execute_approved, propose_with_model
from .profiling import profile_csv
from .storage import audit, initialize, record_upload, write_omop_staging
from .validation import validate_output

@asynccontextmanager
async def lifespan(application: FastAPI):
    initialize()
    yield


app = FastAPI(title="Clinical Partner Data Onboarding", version="0.1.0", lifespan=lifespan)
app.state.sessions = {}


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/uploads")
async def upload_csv(file: UploadFile = File(...), partner_name: str = Form(...)) -> dict:
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(400, "Upload a CSV file")
    max_size = int(os.getenv("MAX_UPLOAD_MB", "10")) * 1024 * 1024
    content = await file.read(max_size + 1)
    if len(content) > max_size:
        raise HTTPException(413, "Upload exceeds configured size limit")
    try:
        frame, profile = profile_csv(content)
    except Exception as exc:
        raise HTTPException(400, f"Could not read CSV: {exc}") from exc
    upload_id = str(uuid.uuid4())
    app.state.sessions[upload_id] = {"frame": frame, "profile": profile, "partner": partner_name,
        "filename": file.filename, "proposals": [], "approved": False}
    record_upload(upload_id, partner_name, file.filename, profile)
    audit(upload_id, "uploaded", {"row_count": len(frame), "column_names": list(frame.columns)})
    # Preview values are only returned to the local user, never sent to the model provider.
    return {"upload_id": upload_id, "profile": profile, "preview": frame.head(20).fillna("").to_dict(orient="records")}


@app.post("/uploads/{upload_id}/mappings")
def mappings(upload_id: str, use_ai: bool = True) -> dict:
    session = _session(upload_id)
    frame, profile = session["frame"], session["profile"]
    proposals = deterministic_proposals(list(frame.columns))
    ai_assisted = False
    if use_ai:
        try:
            proposals = propose_with_model(list(frame.columns), profile)
            ai_assisted = True
        except Exception as exc:
            audit(upload_id, "model_error", {"error": str(exc)})
            if not proposals:
                raise HTTPException(503, f"OpenRouter suggestions unavailable: {exc}") from exc
    session["proposals"] = proposals
    session["approved"] = False
    audit(upload_id, "mapping_proposed", {"proposals": [asdict(p) for p in proposals]})
    return {"mappings": [asdict(p) for p in proposals], "ai_assisted": ai_assisted}


@app.post("/uploads/{upload_id}/approve")
def approve(upload_id: str, body: dict[str, Any]) -> dict:
    session = _session(upload_id)
    if not session["proposals"]:
        raise HTTPException(409, "Generate mapping proposals before approving")
    edited = body.get("mappings", [])
    if len(edited) != len(session["proposals"]):
        raise HTTPException(400, "Submit all proposals for review")
    from .mapping import validate_proposal
    approved = []
    for item in edited:
        try:
            p = validate_proposal(item, list(session["frame"].columns))
        except (ValueError, TypeError, KeyError) as exc:
            raise HTTPException(400, str(exc)) from exc
        p.approved = bool(item.get("approved", False))
        approved.append(p)
    if not any(p.approved for p in approved):
        raise HTTPException(400, "Approve at least one mapping")
    session["proposals"] = approved
    session["approved"] = True
    audit(upload_id, "mapping_approved", {"mappings": [asdict(p) for p in approved]})
    return {"approved": True, "approved_count": sum(p.approved for p in approved)}


@app.post("/uploads/{upload_id}/transform")
def transform(upload_id: str, send_to_omop_staging: bool = False) -> dict:
    session = _session(upload_id)
    if not session["approved"]:
        raise HTTPException(403, "Human approval is required before transformation")
    try:
        result = execute_approved(session["frame"], session["proposals"])
        result, errors = validate_output(result)
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc
    session["result"] = result
    session["errors"] = errors
    inserted = 0
    if send_to_omop_staging and not errors:
        try:
            inserted = write_omop_staging(result, session["partner"])
        except Exception as exc:
            raise HTTPException(503, f"OMOP staging export failed: {exc}") from exc
    audit(upload_id, "transformed", {"rows": len(result), "errors": errors, "omop_staged_rows": inserted})
    return {"rows": len(result), "columns": list(result.columns), "preview": result.head(20).fillna("").to_dict(orient="records"),
            "validation_errors": errors, "omop_staged_rows": inserted,
            "download_url": f"/uploads/{upload_id}/download" if not errors else None}


@app.get("/uploads/{upload_id}/download")
def download(upload_id: str) -> Response:
    session = _session(upload_id)
    if not session.get("approved") or "result" not in session:
        raise HTTPException(403, "Approved, validated transformation is not available")
    result, errors = session["result"], session["errors"]
    if errors:
        raise HTTPException(422, "Transformation contains validation errors")
    return Response(result.to_csv(index=False), media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=validated-clinical-data.csv"})


def _session(upload_id: str) -> dict:
    session = app.state.sessions.get(upload_id)
    if session is None:
        raise HTTPException(404, "Upload session not found")
    return session

