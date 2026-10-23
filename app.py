"""Streamlit UI for review and approval; API owns transformations and audit."""
import asyncio
import io
import os
import sys

# The package uses a src/ layout; Streamlit Cloud installs requirements.txt
# without installing this project as a package.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import pandas as pd
import requests
import streamlit as st

API = os.getenv("API_URL", "").rstrip("/")


class LocalResponse:
    def __init__(self, payload=None, *, status_code=200, content=b""):
        self._payload = payload
        self.status_code = status_code
        self.content = content
        self.text = str(payload) if payload is not None else ""
        self.ok = 200 <= status_code < 400

    def json(self):
        return self._payload


def api_request(method, path, *, params=None, data=None, files=None, json_body=None, timeout=60):
    """Use the Compose API locally; call the same handlers in-process on Streamlit Cloud."""
    if API:
        return requests.request(method, f"{API}{path}", params=params, data=data, files=files,
                                json=json_body, timeout=timeout)
    from fastapi import HTTPException
    from starlette.datastructures import UploadFile
    from onboarding import api as local_api

    try:
        parts = path.strip("/").split("/")
        if method == "POST" and path == "/uploads":
            filename, content, _ = files["file"]
            upload = UploadFile(filename=filename, file=io.BytesIO(content))
            payload = asyncio.run(local_api.upload_csv(upload, data["partner_name"]))
            return LocalResponse(payload)
        if len(parts) == 3 and parts[0] == "uploads" and parts[2] == "mappings":
            return LocalResponse(local_api.mappings(parts[1], (params or {}).get("use_ai", "true").lower() == "true"))
        if len(parts) == 3 and parts[0] == "uploads" and parts[2] == "approve":
            return LocalResponse(local_api.approve(parts[1], json_body or {}))
        if len(parts) == 3 and parts[0] == "uploads" and parts[2] == "transform":
            send = (params or {}).get("send_to_omop_staging", "false").lower() == "true"
            return LocalResponse(local_api.transform(parts[1], send))
        if method == "GET" and len(parts) == 3 and parts[0] == "uploads" and parts[2] == "download":
            response = local_api.download(parts[1])
            return LocalResponse(status_code=response.status_code, content=response.body)
        raise ValueError(f"Unsupported local API request: {method} {path}")
    except HTTPException as exc:
        return LocalResponse({"detail": exc.detail}, status_code=exc.status_code)
    except Exception as exc:
        return LocalResponse({"detail": str(exc)}, status_code=500)


def secret(name, default=""):
    value = os.getenv(name)
    if not value:
        try:
            value = st.secrets.get(name, default)
        except Exception:
            value = default
    if value:
        os.environ[name] = str(value)
    return str(value or default)
st.set_page_config(page_title="Clinical Data Onboarding", layout="wide")
st.title("Partner clinical data onboarding")
st.caption("Upload synthetic or approved partner CSV data, review mappings, validate, approve, and export.")

if "upload_id" not in st.session_state:
    st.session_state.upload_id = None
if "mappings" not in st.session_state:
    st.session_state.mappings = []

with st.form("upload"):
    partner = st.text_input("Partner name", value="Synthetic Partner")
    file = st.file_uploader("Partner CSV", type=["csv"])
    submit = st.form_submit_button("Upload and profile")
if submit and file:
    response = api_request("POST", "/uploads", data={"partner_name": partner}, files={"file": (file.name, file.getvalue(), "text/csv")}, timeout=60)
    if response.ok:
        payload = response.json()
        st.session_state.upload_id = payload["upload_id"]
        st.session_state.profile = payload["profile"]
        st.session_state.preview = pd.DataFrame(payload["preview"])
        st.success(f"Profiled {payload['profile']['rows']} rows")
    else:
        st.error(response.text)

if st.session_state.upload_id:
    st.subheader("Source preview")
    st.dataframe(st.session_state.preview, use_container_width=True)
    st.subheader("Profile")
    st.json(st.session_state.profile)
    secret("OPENROUTER_MODEL", "openrouter/free")
    use_ai = st.toggle("Use OpenRouter suggestions", value=bool(secret("OPENROUTER_API_KEY")))
    if st.button("Suggest mappings"):
        resp = api_request("POST", f"/uploads/{st.session_state.upload_id}/mappings", params={"use_ai": str(use_ai).lower()}, timeout=90)
        if resp.ok:
            payload = resp.json()
            st.session_state.mappings = payload["mappings"]
            if payload.get("ai_assisted"):
                st.success("OpenRouter suggestions received. Review them before approval.")
            elif use_ai:
                st.warning("OpenRouter did not respond; showing deterministic header-rule suggestions.")
            else:
                st.info("Showing deterministic header-rule suggestions.")
        else:
            st.error(resp.text)
    if st.session_state.mappings:
        st.subheader("Review and approve each mapping")
        choices = ["person_source_value", "condition_source_value", "condition_start_date"]
        transforms = ["identity", "parse_date", "normalize_code", "uppercase"]
        edited = []
        for i, item in enumerate(st.session_state.mappings):
            cols = st.columns([2, 2, 2, 1])
            item["target_field"] = cols[0].selectbox("Target", choices, index=choices.index(item["target_field"]), key=f"t{i}")
            item["transformation"] = cols[1].selectbox("Transform", transforms, index=transforms.index(item["transformation"]), key=f"x{i}")
            cols[2].write(f"{item['explanation']} · confidence {item['confidence']:.0%}")
            item["approved"] = cols[3].checkbox("Approve", key=f"a{i}")
            edited.append(item)
        if st.button("Save approval"):
            resp = api_request("POST", f"/uploads/{st.session_state.upload_id}/approve", json_body={"mappings": edited}, timeout=60)
            if resp.ok:
                st.success("Mappings approved")
            else:
                st.error(resp.text)
    if st.button("Run approved transformation"):
        resp = api_request("POST", f"/uploads/{st.session_state.upload_id}/transform", timeout=60)
        if resp.ok:
            result = resp.json()
            st.dataframe(pd.DataFrame(result["preview"]), use_container_width=True)
            if result["validation_errors"]:
                st.error(f"{len(result['validation_errors'])} validation error(s)")
                st.json(result["validation_errors"][:100])
            else:
                data = api_request("GET", result["download_url"], timeout=60).content
                st.download_button("Download validated CSV", data, "validated-clinical-data.csv", "text/csv")
        else:
            st.error(resp.text)

