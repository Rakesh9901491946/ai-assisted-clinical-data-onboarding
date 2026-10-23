from __future__ import annotations

import json
import os
from typing import Any


DDL = """
CREATE TABLE IF NOT EXISTS partner_upload (
  upload_id text PRIMARY KEY, partner_name text NOT NULL, filename text NOT NULL,
  profile jsonb NOT NULL, created_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS mapping_audit (
  audit_id bigserial PRIMARY KEY, upload_id text NOT NULL REFERENCES partner_upload(upload_id),
  event_type text NOT NULL, payload jsonb NOT NULL, created_at timestamptz NOT NULL DEFAULT now()
);
"""


def _connect():
    import psycopg
    url = os.getenv("DATABASE_URL")
    if not url:
        return None
    return psycopg.connect(url)


def initialize() -> None:
    conn = _connect()
    if conn is None:
        return
    with conn, conn.cursor() as cur:
        cur.execute(DDL)


def record_upload(upload_id: str, partner: str, filename: str, profile: dict[str, Any]) -> None:
    conn = _connect()
    if conn is None:
        return
    with conn, conn.cursor() as cur:
        cur.execute("INSERT INTO partner_upload(upload_id,partner_name,filename,profile) VALUES (%s,%s,%s,%s) ON CONFLICT(upload_id) DO NOTHING",
                    (upload_id, partner, filename, json.dumps(profile)))


def audit(upload_id: str, event: str, payload: dict[str, Any]) -> None:
    conn = _connect()
    if conn is None:
        return
    with conn, conn.cursor() as cur:
        cur.execute("INSERT INTO mapping_audit(upload_id,event_type,payload) VALUES (%s,%s,%s)",
                    (upload_id, event, json.dumps(payload, default=str)))


def write_omop_staging(frame, partner: str) -> int:
    """Append approved rows to Project 1 raw partner staging, if configured."""
    import psycopg
    url = os.getenv("OMOP_DATABASE_URL")
    if not url:
        raise RuntimeError("OMOP_DATABASE_URL is required for OMOP staging export")
    rows = [(r.get("person_source_value"), r.get("condition_source_value"), r.get("condition_start_date"), partner)
            for r in frame.to_dict(orient="records")]
    with psycopg.connect(url) as conn, conn.cursor() as cur:
        cur.executemany("INSERT INTO raw.partner_omop_staging(person_source_value,condition_source_value,condition_start_date,source_partner) VALUES (%s,%s,%s,%s) ON CONFLICT DO NOTHING", rows)
    return len(rows)

